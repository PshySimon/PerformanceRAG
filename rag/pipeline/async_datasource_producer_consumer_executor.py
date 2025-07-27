import asyncio
import time
from typing import Any, Dict, Iterator, List, Union

from utils.logger import get_logger

from .executor import PipelineExecutor


class AsyncDataSourceProducerConsumerPipelineExecutor:
    """异步数据源-生产者-消费者Pipeline执行器

    支持迭代器数据源，生产者多线程读取数据源
    """

    def __init__(self, config: Dict[str, Any]):
        self.logger = get_logger(__name__)
        self.config = config

        # 创建数据源、生产者和消费者pipeline
        self.datasource_pipeline = PipelineExecutor(config["datasource"])
        self.producer_pipeline = PipelineExecutor(config["producer"])
        self.consumer_pipeline = PipelineExecutor(config["consumer"])

        # 队列配置
        self.batch_size = config["consumer"]["config"]["batch_size"]
        self.max_queue_size = config["consumer"]["config"]["max_queue_size"]
        self.consumer_tasks = config["consumer"]["config"].get("consumer_tasks", 4)
        self.producer_tasks = config["consumer"]["config"].get("producer_tasks", 2)
        self.timeout = config["consumer"]["config"]["timeout"]

        # 流量控制配置
        self.rate_limit_per_second = config["consumer"]["config"].get(
            "rate_limit_per_second", 3
        )
        self.rate_limit_window = 1.0

        # 异步队列和控制变量
        self.data_queue = None
        self.producer_finished = None
        self.consumer_finished = None
        self.monitor_stop = None

        # 生产者协调变量
        self.active_producers = 0
        self.producer_lock = None
        self.data_source_iterator = None
        self.data_source_lock = None

        # 流量控制变量
        self.rate_limiter_lock = None
        self.request_timestamps = []

        # 统计信息
        self.total_files_read = 0  # 新增：已读取文件数
        self.total_data_read = 0
        self.total_data_produced = 0
        self.total_data_fetched = 0
        self.total_data_consumed = 0
        self.stats_lock = None

        # 监控配置
        self.monitor_interval = 5
        self.start_time = None

        self.preload_all_data = config["consumer"]["config"].get(
            "preload_all_data", False
        )
        self.all_data_items = None
        self.total_data_count = 0

    async def _rate_limit_check(self):
        """检查并执行流量控制"""
        async with self.rate_limiter_lock:
            current_time = time.time()

            # 清理过期的时间戳
            self.request_timestamps = [
                ts
                for ts in self.request_timestamps
                if current_time - ts < self.rate_limit_window
            ]

            # 如果当前时间窗口内的请求数已达到限制，需要等待
            if len(self.request_timestamps) >= self.rate_limit_per_second:
                oldest_request = min(self.request_timestamps)
                wait_time = self.rate_limit_window - (current_time - oldest_request)

                if wait_time > 0:
                    self.logger.debug(
                        f"🚦 流量控制：等待 {wait_time:.2f}s 以避免超过限速 {self.rate_limit_per_second}/s"
                    )
                    await asyncio.sleep(wait_time)
                    current_time = time.time()

            # 记录当前请求时间戳
            self.request_timestamps.append(current_time)

    def _create_data_source_iterator(
        self, input_data: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """创建数据源迭代器"""
        try:
            # 确保数据源pipeline已构建
            if not self.datasource_pipeline._built:
                self.datasource_pipeline.build()

            # 使用数据源pipeline的流式执行
            return self.datasource_pipeline.run_stream(input_data)

        except Exception as e:
            self.logger.error(f"创建数据源迭代器失败: {e}")
            raise

    async def _get_next_data_item(self) -> Union[Dict[str, Any], None]:
        """从数据源迭代器获取下一个数据项（线程安全）"""
        async with self.data_source_lock:
            try:
                data_item = next(self.data_source_iterator)
                # 统计读取的文件数
                async with self.stats_lock:
                    self.total_files_read += 1
                    self.total_data_read += len(data_item.get("documents", []))
                return data_item
            except StopIteration:
                return None
            except Exception as e:
                self.logger.error(f"从数据源读取数据失败: {e}")
                return None

    async def _producer_worker(self, worker_id: int):
        """异步生产者工作任务"""
        try:
            self.logger.info(
                f"🚀 异步生产者任务 {worker_id} 开始工作...（流量限制: {self.rate_limit_per_second}/s）"
            )

            # 注册活跃生产者
            async with self.producer_lock:
                self.active_producers += 1

            # 确保生产者pipeline已构建
            if not self.producer_pipeline._built:
                self.producer_pipeline.build()

            # 持续从数据源读取数据
            while True:
                # 从数据源获取下一个数据项
                data_item = await self._get_next_data_item()
                if data_item is None:
                    break  # 数据源已耗尽

                try:
                    # 执行流量控制检查
                    await self._rate_limit_check()

                    # 执行生产者pipeline
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None, lambda: list(self.producer_pipeline.run_stream(data_item))
                    )

                    # 将结果放入异步队列
                    for result in results:
                        await asyncio.wait_for(
                            self.data_queue.put(result), timeout=self.timeout
                        )

                        # 统计放入队列的item数量
                        async with self.stats_lock:
                            self.total_data_produced += len(result.get("documents", []))

                except Exception as e:
                    import traceback

                    self.logger.error(
                        f"生产者任务 {worker_id} 处理数据项失败: {type(e).__name__}: {str(e)}\n"
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
                            self.total_data_fetched += len(item.get("documents", []))

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

            # 在线程池中执行消费者pipeline
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.consumer_pipeline.run, consumer_input)

            # 统计消费数量
            async with self.stats_lock:
                self.total_data_consumed += sum(
                    len(item.get("documents", [])) for item in batch_data
                )

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
                    files_per_sec = (
                        self.total_files_read / elapsed if elapsed > 0 else 0
                    )
                    produced_per_sec = (
                        self.total_data_produced / elapsed if elapsed > 0 else 0
                    )
                    consume_per_sec = (
                        self.total_data_consumed / elapsed if elapsed > 0 else 0
                    )

                    processing_count = (
                        self.total_data_fetched - self.total_data_consumed
                    )

                    if hasattr(self, 'total_data_count') and self.total_data_count:
                        # 预加载模式：显示精确进度和剩余时间估算
                        progress = (self.total_files_read / self.total_data_count) * 100
                        remaining_files = self.total_data_count - self.total_files_read
                        
                        # 估算剩余时间
                        if files_per_sec > 0:
                            eta_seconds = remaining_files / files_per_sec
                            eta_minutes = eta_seconds / 60
                            if eta_minutes > 60:
                                eta_str = f"{eta_minutes/60:.1f}h"
                            elif eta_minutes > 1:
                                eta_str = f"{eta_minutes:.1f}m"
                            else:
                                eta_str = f"{eta_seconds:.0f}s"
                        else:
                            eta_str = "未知"
                        
                        if (
                            self.producer_finished.is_set()
                            and self.data_queue.qsize() == 0
                        ):
                            status = "收尾中"
                        else:
                            status = "处理中"
                        
                        self.logger.info(
                            f"📊 [{status}] 进度: {progress:.1f}% ({self.total_files_read}/{self.total_data_count}) | "
                            f"速度: {files_per_sec:.1f}文件/s,生产{produced_per_sec:.1f}条/s, 消费{consume_per_sec:.1f}条/s | "
                            f"队列: {processing_count}待处理 | ETA: {eta_str}"
                        )
                    else:
                        # 流式模式：显示已处理数量和速度
                        if (
                            self.producer_finished.is_set()
                            and self.data_queue.qsize() == 0
                        ):
                            status = "收尾中"
                        else:
                            status = "处理中"
                        
                        self.logger.info(
                            f"📊 [{status}] 已处理: {self.total_files_read} 个文件 | "
                            f"速度: {files_per_sec:.1f}文件/s, {consume_per_sec:.1f}条/s | "
                            f"队列: {processing_count}待处理 | 耗时: {elapsed:.1f}s"
                        )

            await asyncio.sleep(self.monitor_interval)

    def _preload_all_data(self, input_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """预加载所有数据到内存"""
        self.logger.info("📋 预加载所有数据到内存...")

        # 创建数据源迭代器
        data_iterator = self._create_data_source_iterator(input_data)

        # 将所有数据加载到内存
        all_items = []
        for item in data_iterator:
            all_items.append(item)

        self.total_data_count = len(all_items)
        self.logger.info(f"📋 预加载完成，总数据量: {self.total_data_count} 个数据项")

        return all_items

    async def run_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """异步执行数据源-生产者-消费者pipeline"""
        self.logger.info(
            f"🚀 启动异步数据源-生产者-消费者Pipeline（生产者任务数: {self.producer_tasks}，消费者任务数: {self.consumer_tasks}，流量限制: {self.rate_limit_per_second}/s）..."
        )
        self.start_time = time.time()

        # 初始化异步控制变量
        self.data_queue = asyncio.Queue(maxsize=self.max_queue_size)
        self.producer_finished = asyncio.Event()
        self.consumer_finished = asyncio.Event()
        self.monitor_stop = asyncio.Event()
        self.producer_lock = asyncio.Lock()
        self.data_source_lock = asyncio.Lock()
        self.stats_lock = asyncio.Lock()
        self.rate_limiter_lock = asyncio.Lock()

        try:
            # 创建数据源迭代器
            self.logger.info("📋 创建数据源迭代器...")
            if self.preload_all_data:
                # 预加载模式：一次性加载所有数据到内存
                self.all_data_items = self._preload_all_data(input_data)
                self.data_source_iterator = iter(self.all_data_items)
            else:
                # 流式模式：使用迭代器
                self.data_source_iterator = self._create_data_source_iterator(
                    input_data
                )
                self.total_data_count = None
            # 创建所有异步任务
            tasks = []

            # 启动进度监控任务
            monitor_task = asyncio.create_task(self._progress_monitor())
            tasks.append(monitor_task)

            # 启动生产者任务
            producer_tasks = []
            for i in range(self.producer_tasks):
                task = asyncio.create_task(self._producer_worker(i))
                producer_tasks.append(task)
                tasks.append(task)
                self.logger.info(f"📤 启动异步生产者任务 {i}")

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
            if "monitor_task" in locals():
                await monitor_task

            # 最终统计
            total_time = time.time() - self.start_time
            avg_files_per_sec = (
                self.total_files_read / total_time if total_time > 0 else 0
            )
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
                f"🎯 异步数据源Pipeline执行完成！\n"
                f"   📁 读取文件: {self.total_files_read} 个 (平均 {avg_files_per_sec:.1f}/s)\n"
                f"   📖 读取数据: {self.total_data_read} 条 (平均 {avg_read_per_sec:.1f}/s)\n"
                f"   📤 生产数据: {self.total_data_produced} 条 (平均 {avg_produce_per_sec:.1f}/s)\n"
                f"   📥 消费数据: {self.total_data_consumed} 条 (平均 {avg_consume_per_sec:.1f}/s)\n"
                f"   ⏱️  总耗时: {total_time:.1f}s"
            )

        return {
            "status": "completed",
            "total_files_read": self.total_files_read,
            "total_data_read": self.total_data_read,
            "total_data_produced": self.total_data_produced,
            "total_data_consumed": self.total_data_consumed,
            "total_time": total_time,
        }

    def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """同步接口，内部调用异步实现"""
        return asyncio.run(self.run_async(input_data))
