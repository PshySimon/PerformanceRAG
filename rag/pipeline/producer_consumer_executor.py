import math
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from utils.logger import get_logger

from .executor import PipelineExecutor


class ProducerConsumerPipelineExecutor:
    """生产者-消费者Pipeline执行器"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger(__name__)
        self.config = config

        # 创建生产者和消费者pipeline
        self.producer_pipeline = PipelineExecutor(config["producer"])
        self.consumer_pipeline = PipelineExecutor(config["consumer"])

        # 队列配置
        self.batch_size = config["consumer"]["config"]["batch_size"]
        self.max_queue_size = config["consumer"]["config"]["max_queue_size"]
        self.consumer_threads = config["consumer"]["config"]["consumer_threads"]
        # 新增：生产者线程数配置
        self.producer_threads = config["consumer"]["config"].get("producer_threads", 1)
        self.timeout = config["consumer"]["config"]["timeout"]

        # 创建队列和控制变量
        self.data_queue = queue.Queue(maxsize=self.max_queue_size)
        self.producer_finished = threading.Event()
        self.consumer_finished = threading.Event()
        self.monitor_stop = threading.Event()

        # 新增：生产者协调变量
        self.active_producers = 0
        self.producer_lock = threading.Lock()
        self.data_source = None  # 存储数据源用于分片

        # 统计信息 - 明确区分读取、生产、消费
        self.total_data_read = 0  # 已读取数据条数（从数据源读取的原始数据）
        self.total_data_produced = 0  # 已生产数据条数（生产者处理后放入队列的数据）
        self.total_data_fetched = 0  # 已取出数据条数（消费者从队列取出的数据）
        self.total_data_consumed = 0  # 已消费数据条数（消费者处理完成的数据）
        self.stats_lock = threading.Lock()

        # 监控配置
        self.monitor_interval = 5  # 每5秒显示一次进度
        self.start_time = None

    def _get_data_source_items(self, input_data: Dict[str, Any]) -> List[Any]:
        """获取数据源的所有项目，用于分片"""
        try:
            # 确保pipeline已构建
            if not self.producer_pipeline._built:
                self.producer_pipeline.build()

            # 从入口组件获取数据源
            if not self.producer_pipeline.entry_point:
                raise RuntimeError("无法获取生产者pipeline的入口组件")

            # 获取数据源项目列表（这里需要根据具体的loader实现来调整）
            entry_component = self.producer_pipeline.entry_point

            # 如果是文件loader，获取文件列表
            if hasattr(entry_component, "get_file_list"):
                return entry_component.get_file_list(input_data)
            elif hasattr(entry_component, "get_data_items"):
                return entry_component.get_data_items(input_data)
            else:
                # 回退方案：通过get_data_length估算
                total_count = entry_component.get_data_length(input_data)
                return list(range(total_count))  # 使用索引作为分片依据

        except Exception as e:
            self.logger.error(f"获取数据源项目失败: {e}")
            raise

    def _create_data_chunks(
        self, data_items: List[Any], num_chunks: int
    ) -> List[List[Any]]:
        """将数据源分片给多个生产者线程"""
        if not data_items:
            return []

        chunk_size = math.ceil(len(data_items) / num_chunks)
        chunks = []

        for i in range(0, len(data_items), chunk_size):
            chunk = data_items[i : i + chunk_size]
            if chunk:  # 确保chunk不为空
                chunks.append(chunk)

        return chunks

    def _producer_worker(
        self, worker_id: int, data_chunk: List[Any], input_data: Dict[str, Any]
    ):
        """生产者工作线程（多线程版本）"""
        try:
            self.logger.info(
                f"🚀 生产者线程 {worker_id} 开始工作，处理 {len(data_chunk)} 个数据项..."
            )

            # 注册活跃生产者
            with self.producer_lock:
                self.active_producers += 1

            # 确保pipeline已构建
            if not self.producer_pipeline._built:
                self.producer_pipeline.build()

            # 为每个数据项创建独立的输入
            for data_item in data_chunk:
                try:
                    # 根据数据项类型创建输入
                    if isinstance(data_item, str):  # 文件路径
                        item_input = {**input_data, "path": data_item}
                    elif isinstance(data_item, dict):  # 已经是输入数据
                        item_input = data_item
                    else:  # 其他类型，使用索引
                        item_input = {**input_data, "item_index": data_item}

                    # 执行生产者pipeline并统计放入队列的item数量
                    for result in self.producer_pipeline.run_stream(item_input):
                        # 将结果放入队列
                        self.data_queue.put(result, timeout=self.timeout)

                        # 统计放入队列的item数量
                        with self.stats_lock:
                            self.total_data_produced += 1

                except Exception as e:
                    self.logger.error(f"生产者线程 {worker_id} 处理数据项失败: {e}")
                    continue

            self.logger.info(f"✅ 生产者线程 {worker_id} 完成")

        except Exception as e:
            self.logger.error(f"❌ 生产者线程 {worker_id} 异常: {e}")
            raise
        finally:
            # 注销活跃生产者
            with self.producer_lock:
                self.active_producers -= 1
                if self.active_producers == 0:
                    self.producer_finished.set()
                    self.logger.info("✅ 所有生产者线程已完成")

    def _consumer_worker(self, worker_id: int):
        """消费者工作线程"""
        try:
            self.logger.info(f"🔧 消费者线程 {worker_id} 开始工作...")

            while True:
                batch_data = []

                # 批量从队列中取数据
                for _ in range(self.batch_size):
                    try:
                        item = self.data_queue.get(timeout=self.timeout)
                        batch_data.append(item)

                        # 统计取出的数据
                        with self.stats_lock:
                            self.total_data_fetched += 1

                    except queue.Empty:
                        if self.producer_finished.is_set() and self.data_queue.empty():
                            if batch_data:
                                self._process_batch(batch_data, worker_id)
                            return
                        break

                # 处理当前批次数据
                if batch_data:
                    self._process_batch(batch_data, worker_id)

                # 检查退出条件
                if self.producer_finished.is_set() and self.data_queue.empty():
                    return

        except Exception as e:
            self.logger.error(f"❌ 消费者线程 {worker_id} 异常: {e}")

    def _process_batch(self, batch_data: List[Dict[str, Any]], worker_id: int):
        """处理一个批次的数据"""
        try:
            # 合并批次数据
            merged_documents = []
            for item in batch_data:
                if "documents" in item:
                    merged_documents.extend(item["documents"])

            if not merged_documents:
                return

            # 记录开始处理的文档数
            doc_count = len(merged_documents)

            # 构造消费者输入
            consumer_input = {
                "documents": merged_documents,
                "batch_size": doc_count,
                "worker_id": worker_id,
            }

            # 执行消费者pipeline
            self.consumer_pipeline.run(consumer_input)

            # 只有在真正处理完成后才统计消费数量
            with self.stats_lock:
                self.total_data_consumed += len(batch_data)

        except Exception as e:
            self.logger.error(f"❌ 消费者线程 {worker_id} 处理批次异常: {e}")

    def _progress_monitor(self):
        """实时进度监控线程"""
        while not self.monitor_stop.is_set():
            if self.start_time:
                with self.stats_lock:
                    current_time = time.time()
                    elapsed = current_time - self.start_time

                    # 计算处理速度
                    produced_per_sec = (
                        self.total_data_produced / elapsed if elapsed > 0 else 0
                    )
                    fetched_per_sec = (
                        self.total_data_fetched / elapsed if elapsed > 0 else 0
                    )
                    consume_per_sec = (
                        self.total_data_consumed / elapsed if elapsed > 0 else 0
                    )

                    processing_count = (
                        self.total_data_fetched - self.total_data_consumed
                    )

                    if self.total_data_consumed > 0:
                        if (
                            self.producer_finished.is_set()
                            and self.data_queue.qsize() == 0
                        ):
                            status = "收尾中"
                        else:
                            status = "处理中"

                        self.logger.info(
                            f"📊 {status} | 数据源: {self.total_data_read}"
                            f"| 已生产: {self.total_data_produced} ({produced_per_sec:.1f}/s) "
                            f"| 已取出: {self.total_data_fetched} ({fetched_per_sec:.1f}/s) "
                            f"| 处理中: {processing_count} "
                            f"| 已完成: {self.total_data_consumed} ({consume_per_sec:.1f}/s) "
                            f"| 队列: {self.data_queue.qsize()}"
                        )

            self.monitor_stop.wait(self.monitor_interval)

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """执行生产者-消费者pipeline"""
        self.logger.info(
            f"🚀 启动生产者-消费者Pipeline（生产者线程数: {self.producer_threads}）..."
        )
        self.start_time = time.time()
        monitor_thread = None  # 初始化monitor_thread变量

        try:
            # 获取数据源并分片
            self.logger.info("📋 准备数据源分片...")
            data_items = self._get_data_source_items(input_data)

            with self.stats_lock:
                self.total_data_read = len(data_items)

            self.logger.info(
                f"📋 数据源总量: {len(data_items)}，将分配给 {self.producer_threads} 个生产者线程"
            )

            # 创建数据分片
            data_chunks = self._create_data_chunks(data_items, self.producer_threads)

            if not data_chunks:
                self.logger.warning("⚠️ 没有数据需要处理")
                return {
                    "status": "completed",
                    "total_data_read": 0,
                    "total_data_produced": 0,
                    "total_data_consumed": 0,
                    "total_time": 0,
                }

            # 启动实时进度监控线程
            monitor_thread = threading.Thread(
                target=self._progress_monitor, name="ProgressMonitor"
            )
            monitor_thread.start()

            # 启动多个生产者线程
            with ThreadPoolExecutor(
                max_workers=self.producer_threads, thread_name_prefix="Producer"
            ) as producer_executor:
                producer_futures = []

                for i, chunk in enumerate(data_chunks):
                    if chunk:  # 确保chunk不为空
                        future = producer_executor.submit(
                            self._producer_worker, i, chunk, input_data
                        )
                        producer_futures.append(future)
                        self.logger.info(
                            f"📤 启动生产者线程 {i}，处理 {len(chunk)} 个数据项"
                        )

                # 启动消费者线程池
                with ThreadPoolExecutor(
                    max_workers=self.consumer_threads, thread_name_prefix="Consumer"
                ) as consumer_executor:
                    consumer_futures = []
                    for i in range(self.consumer_threads):
                        future = consumer_executor.submit(self._consumer_worker, i)
                        consumer_futures.append(future)

                    # 等待所有生产者完成
                    for future in producer_futures:
                        future.result()

                    self.logger.info("✅ 所有生产者线程已完成")

                    # 等待所有消费者完成
                    for future in consumer_futures:
                        future.result()

                    self.logger.info("✅ 所有消费者线程已完成")

        finally:
            # 等待消费者完全完成后再停止监控
            time.sleep(0.1)  # 给消费者一点时间完成最后的统计

            # 最后一次进度更新
            with self.stats_lock:
                current_time = time.time()
                elapsed = current_time - self.start_time
                consume_per_sec = (
                    self.total_data_consumed / elapsed if elapsed > 0 else 0
                )

                self.logger.info(
                    f"📊 数据源: {self.total_data_read} "
                    f"| 已生产数据: {self.total_data_produced} "
                    f"| 已消费数据: {self.total_data_consumed} ({consume_per_sec:.1f}/s) "
                    f"| 队列: {self.data_queue.qsize()}"
                )

            # 停止监控线程
            self.monitor_stop.set()
            if monitor_thread is not None:  # 添加空值检查
                monitor_thread.join()

            # 最终统计
            total_time = time.time() - self.start_time
            avg_read_per_sec = (
                self.total_data_read / total_time if total_time > 0 else 0
            )
            avg_produce_per_sec = (
                self.total_data_produced / total_time if total_time > 0 else 0
            )
            avg_consume_per_sec = (
                self.total_data_consumed / total_time if total_time > 0 else 0
            )

            self.logger.info(
                f"🎯 Pipeline执行完成！\n"
                f"   📖 读取数据: {self.total_data_read} 条 (平均 {avg_read_per_sec:.1f}/s)\n"
                f"   📤 生产数据: {self.total_data_produced} 条 (平均 {avg_produce_per_sec:.1f}/s)\n"
                f"   📥 消费数据: {self.total_data_consumed} 条 (平均 {avg_consume_per_sec:.1f}/s)\n"
                f"   ⏱️  总耗时: {total_time:.1f}s"
            )

        return {
            "status": "completed",
            "total_data_read": self.total_data_read,
            "total_data_produced": self.total_data_produced,
            "total_data_consumed": self.total_data_consumed,
            "total_time": total_time,
        }
