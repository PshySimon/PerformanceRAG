import asyncio
import math
import time
from typing import Any, Dict, List

from utils.logger import get_logger

from .executor import PipelineExecutor


class AsyncProducerConsumerPipelineExecutor:
    """异步生产者-消费者Pipeline执行器"""

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger(__name__)
        self.config = config

        # 创建生产者和消费者pipeline
        self.producer_pipeline = PipelineExecutor(config["producer"])
        self.consumer_pipeline = PipelineExecutor(config["consumer"])

        # 队列配置
        self.batch_size = config["consumer"]["config"]["batch_size"]
        self.max_queue_size = config["consumer"]["config"]["max_queue_size"]
        self.consumer_tasks = config["consumer"]["config"].get("consumer_tasks", 4)
        self.producer_tasks = config["consumer"]["config"].get("producer_tasks", 2)
        self.timeout = config["consumer"]["config"]["timeout"]

        # 流量控制配置
        self.rate_limit_per_second = config["consumer"]["config"].get("rate_limit_per_second", 3)  # 每秒最大请求数
        self.rate_limit_window = 1.0  # 时间窗口（秒）
        
        # 异步队列和控制变量
        self.data_queue = None  # 将在运行时创建
        self.producer_finished = None
        self.consumer_finished = None
        self.monitor_stop = None

        # 生产者协调变量
        self.active_producers = 0
        self.producer_lock = None
        self.data_source = None

        # 流量控制变量
        self.rate_limiter_lock = None
        self.request_timestamps = []  # 记录请求时间戳

        # 统计信息
        self.total_data_read = 0
        self.total_data_produced = 0
        self.total_data_fetched = 0
        self.total_data_consumed = 0
        self.stats_lock = None

        # 监控配置
        self.monitor_interval = 5
        self.start_time = None

    async def _rate_limit_check(self):
        """检查并执行流量控制"""
        async with self.rate_limiter_lock:
            current_time = time.time()
            
            # 清理过期的时间戳（超过时间窗口的）
            self.request_timestamps = [
                ts for ts in self.request_timestamps 
                if current_time - ts < self.rate_limit_window
            ]
            
            # 如果当前时间窗口内的请求数已达到限制，需要等待
            if len(self.request_timestamps) >= self.rate_limit_per_second:
                # 计算需要等待的时间
                oldest_request = min(self.request_timestamps)
                wait_time = self.rate_limit_window - (current_time - oldest_request)
                
                if wait_time > 0:
                    self.logger.debug(f"🚦 流量控制：等待 {wait_time:.2f}s 以避免超过限速 {self.rate_limit_per_second}/s")
                    await asyncio.sleep(wait_time)
                    current_time = time.time()
            
            # 记录当前请求时间戳
            self.request_timestamps.append(current_time)

    def _get_data_source_items(self, input_data: Dict[str, Any]) -> List[Any]:
        """获取数据源的所有项目，用于分片"""
        try:
            # 确保pipeline已构建
            if not self.producer_pipeline._built:
                self.producer_pipeline.build()

            # 从入口组件获取数据源
            if not self.producer_pipeline.entry_point:
                raise RuntimeError("无法获取生产者pipeline的入口组件")

            entry_component = self.producer_pipeline.entry_point

            # 如果是文件loader，获取文件列表
            if hasattr(entry_component, "get_file_list"):
                return entry_component.get_file_list(input_data)
            elif hasattr(entry_component, "get_data_items"):
                return entry_component.get_data_items(input_data)
            else:
                # 回退方案：通过get_data_length估算
                total_count = entry_component.get_data_length(input_data)
                return list(range(total_count))

        except Exception as e:
            self.logger.error(f"获取数据源项目失败: {e}")
            raise

    def _create_data_chunks(
        self, data_items: List[Any], num_chunks: int
    ) -> List[List[Any]]:
        """将数据源分片给多个生产者任务"""
        if not data_items:
            return []

        chunk_size = math.ceil(len(data_items) / num_chunks)
        chunks = []

        for i in range(0, len(data_items), chunk_size):
            chunk = data_items[i : i + chunk_size]
            if chunk:
                chunks.append(chunk)

        return chunks

    async def _producer_worker(
        self, worker_id: int, data_chunk: List[Any], input_data: Dict[str, Any]
    ):
        """异步生产者工作任务"""
        try:
            self.logger.info(
                f"🚀 异步生产者任务 {worker_id} 开始工作，处理 {len(data_chunk)} 个数据项...（流量限制: {self.rate_limit_per_second}/s）"
            )

            # 注册活跃生产者
            async with self.producer_lock:
                self.active_producers += 1

            # 确保pipeline已构建
            if not self.producer_pipeline._built:
                self.producer_pipeline.build()

            # 为每个数据项创建独立的输入
            for data_item in data_chunk:
                try:
                    # 执行流量控制检查
                    await self._rate_limit_check()
                    
                    # 根据数据项类型创建输入
                    if isinstance(data_item, str):  # 文件路径
                        item_input = {**input_data, "path": data_item}
                    elif isinstance(data_item, dict):  # 已经是输入数据
                        item_input = data_item
                    else:  # 其他类型，使用索引
                        item_input = {**input_data, "item_index": data_item}

                    # 执行生产者pipeline并统计放入队列的item数量
                    for result in self.producer_pipeline.run_stream(item_input):
                        # 将结果放入异步队列
                        await asyncio.wait_for(
                            self.data_queue.put(result), timeout=self.timeout
                        )

                        # 统计放入队列的item数量
                        async with self.stats_lock:
                            self.total_data_produced += len(result["documents"])

                except Exception as e:
                    # 改进错误日志，提供更详细的信息
                    import traceback
                    error_details = {
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                        "data_item": str(data_item)[:100] if data_item else "None",
                        "worker_id": worker_id
                    }
                    self.logger.error(
                        f"生产者任务 {worker_id} 处理数据项失败: {error_details['error_type']}: {error_details['error_message']}\n"
                        f"数据项: {error_details['data_item']}\n"
                        f"完整错误堆栈: {traceback.format_exc()}"
                    )
                    continue

            self.logger.info(f"✅ 异步生产者任务 {worker_id} 完成")

        except Exception as e:
            self.logger.error(f"❌ 异步生产者任务 {worker_id} 异常: {e}")
            raise
        finally:
            # 注销活跃生产者
            async with self.producer_lock:
                self.active_producers -= 1
                if self.active_producers == 0:
                    self.producer_finished.set()
                    self.logger.info("✅ 所有异步生产者任务已完成")

    async def _consumer_worker(self, worker_id: int):
        """异步消费者工作任务"""
        try:
            self.logger.info(f"🔧 异步消费者任务 {worker_id} 开始工作...")

            while True:
                batch_data = []

                # 批量从队列中取数据
                for _ in range(self.batch_size):
                    try:
                        item = await asyncio.wait_for(
                            self.data_queue.get(), timeout=self.timeout
                        )
                        batch_data.append(item)

                        # 统计取出的数据
                        async with self.stats_lock:
                            self.total_data_fetched += len(item["documents"])

                    except asyncio.TimeoutError:
                        if self.producer_finished.is_set() and self.data_queue.empty():
                            if batch_data:
                                await self._process_batch(batch_data, worker_id)
                            return
                        break

                # 处理当前批次数据
                if batch_data:
                    await self._process_batch(batch_data, worker_id)

                # 检查退出条件
                if self.producer_finished.is_set() and self.data_queue.empty():
                    return

        except Exception as e:
            self.logger.error(f"❌ 异步消费者任务 {worker_id} 异常: {e}")

    async def _process_batch(self, batch_data: List[Dict[str, Any]], worker_id: int):
        """异步处理一个批次的数据"""
        try:
            # 合并批次数据
            merged_documents = []
            for item in batch_data:
                if "documents" in item:
                    merged_documents.extend(item["documents"])

            if not merged_documents:
                return

            # 构造消费者输入
            consumer_input = {
                "documents": merged_documents,
                "batch_size": len(merged_documents),
                "worker_id": worker_id,
            }

            # 在线程池中执行消费者pipeline（因为pipeline本身是同步的）
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None, self.consumer_pipeline.run, consumer_input
            )

            # 只有在真正处理完成后才统计消费数量
            async with self.stats_lock:
                self.total_data_consumed += sum(len(item["documents"]) for item in batch_data)

        except Exception as e:
            self.logger.error(f"❌ 异步消费者任务 {worker_id} 处理批次异常: {e}")

    async def _progress_monitor(self):
        """异步实时进度监控任务"""
        while not self.monitor_stop.is_set():
            if self.start_time:
                async with self.stats_lock:
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
                            f"📊 [异步] {status} | 数据源: {self.total_data_read}"
                            f"| 已生产: {self.total_data_produced} ({produced_per_sec:.1f}/s) "
                            f"| 已取出: {self.total_data_fetched} ({fetched_per_sec:.1f}/s) "
                            f"| 处理中: {processing_count} "
                            f"| 已完成: {self.total_data_consumed} ({consume_per_sec:.1f}/s) "
                            f"| 队列: {self.data_queue.qsize()}"
                        )

            await asyncio.sleep(self.monitor_interval)

    async def run_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行生产者-消费者pipeline"""
        self.logger.info(
            f"🚀 启动异步生产者-消费者Pipeline（生产者任务数: {self.producer_tasks}，消费者任务数: {self.consumer_tasks}，流量限制: {self.rate_limit_per_second}/s）..."
        )
        self.start_time = time.time()

        # 初始化异步控制变量
        self.data_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.producer_finished = asyncio.Event()
        self.consumer_finished = asyncio.Event()
        self.monitor_stop = asyncio.Event()
        self.producer_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()
        self.rate_limiter_lock = asyncio.Lock()  # 新增流量控制锁

        try:
            # 获取数据源并分片
            self.logger.info("📋 准备数据源分片...")
            data_items = self._get_data_source_items(input_data)

            async with self.stats_lock:
                self.total_data_read = len(data_items)

            self.logger.info(
                f"📋 数据源总量: {len(data_items)}，将分配给 {self.producer_tasks} 个异步生产者任务"
            )

            # 创建数据分片
            data_chunks = self._create_data_chunks(data_items, self.producer_tasks)

            if not data_chunks:
                self.logger.warning("⚠️ 没有数据需要处理")
                return {
                    "status": "completed",
                    "total_data_read": 0,
                    "total_data_produced": 0,
                    "total_data_consumed": 0,
                    "total_time": 0,
                }

            # 创建所有异步任务
            tasks = []

            # 启动进度监控任务
            monitor_task = asyncio.create_task(self._progress_monitor())
            tasks.append(monitor_task)

            # 启动生产者任务
            producer_tasks = []
            for i, chunk in enumerate(data_chunks):
                if chunk:
                    task = asyncio.create_task(
                        self._producer_worker(i, chunk, input_data)
                    )
                    producer_tasks.append(task)
                    tasks.append(task)
                    self.logger.info(
                        f"📤 启动异步生产者任务 {i}，处理 {len(chunk)} 个数据项"
                    )

            # 启动消费者任务
            consumer_tasks = []
            for i in range(self.consumer_tasks):
                task = asyncio.create_task(self._consumer_worker(i))
                consumer_tasks.append(task)
                tasks.append(task)

            # 等待所有生产者完成
            await asyncio.gather(*producer_tasks)
            self.logger.info("✅ 所有异步生产者任务已完成")

            # 等待所有消费者完成
            await asyncio.gather(*consumer_tasks)
            self.logger.info("✅ 所有异步消费者任务已完成")

        finally:
            # 停止监控任务
            self.monitor_stop.set()
            if 'monitor_task' in locals():
                await monitor_task

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
                f"🎯 异步Pipeline执行完成！\n"
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

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """同步接口，内部调用异步实现"""
        return asyncio.run(self.run_async(input_data))