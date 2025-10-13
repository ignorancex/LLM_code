import zmq
from dataclasses import dataclass,field
import datetime
import uuid
from collections import defaultdict
from typing import Optional,Any
import numpy as np
import pickle
import os
import time
from .utils import logger,_process_inputs,Timer
import threading
import queue
from transformers import AutoProcessor
import socket
import random
from urllib.parse import urlparse

@dataclass
class TaskStatus:
    task_id: int
    completion_id: uuid.UUID
    score: float
    created_time: datetime.datetime = field(default_factory=datetime.datetime.now)
    advantage: Optional[float] = None
    
@dataclass
class TaskAndContent:
    data: dict
    status: TaskStatus
    
@dataclass
class SyncAdvantagesRequest:
    gid: int

DEFAULT_THRESHOLD = 0.90

def global_sync_proc(
    *args,**kwargs
):
    """Manage Global Task Synchronization Signal And Assign Advantages."""
    manager = GlobalSyncManager(*args,**kwargs)
    manager.start()
    
def local_balance_proc(
    *args,**kwargs
):
    """Balance data and tasks created in local machine, sync with global."""
    manager = LocalBalanceManager(*args,**kwargs)
    manager.start()
    
class GlobalSyncManager:
    """管理全局任务同步信号并分配优势值。"""
    
    def __init__(
        self,
        sync_address: str,
        collect_address: str,
        num_generations: int,
        num_to_sync: int,
        num_nodes: int,
        tp_size: int = 1
    ):
        """初始化全局同步管理器。
        
        Args:
            sync_address: 同步信号地址
            collect_address: 结果收集地址
            num_generations: 每个任务的生成数量
            num_to_sync: 需要同步的数据量
            num_nodes: 节点数量
            tp_size: 张量并行大小
        """
        self.sync_address = sync_address
        self.collect_address = collect_address
        self.num_generations = num_generations
        self.num_to_sync = num_to_sync
        self.num_nodes = num_nodes
        self.tp_size = tp_size
        
        # 初始化变量
        self.num_machines = int(os.environ.get("NUM_MACHINES", "1"))
        self.ack_advantages = 0
        self.total_ack = 0
        self.recv_count = 0
        self.send_count = 0
        self.current_gid = 0
        self.sync_steps = 0
        self.sync_pool = {}
        self.sync_count = defaultdict(int)
        self.task_collection = defaultdict(list)
        self.node_queue_lengths = {}
        
        # 初始化ZMQ
        self.zmqctx = zmq.Context(self.num_machines*2)
        self.sync_lock = threading.Lock()
        
        # 设置同步发送器
        self.sync_sender = self.zmqctx.socket(zmq.PUB)
        self.sync_sender.setsockopt(zmq.TCP_KEEPALIVE, 1)
        self.sync_sender.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        self.sync_sender.setsockopt(zmq.TCP_KEEPALIVE_CNT, 5)
        self.sync_sender.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 10)
        self.sync_sender.bind(self.sync_address)
        
        # 设置任务收集器
        self.task_collect = self.zmqctx.socket(zmq.REP)
        self.task_collect.bind(self.collect_address)
    
    def _monitor(self):
        """监控线程，定期报告状态"""
        interval = int(os.environ.get("MONITOR_INTERVAL", "30"))
        while True:
            last_count = self.recv_count
            time.sleep(interval)
            logger.info(f"[ Global GID: {self.current_gid} | SyncPool Size: {len(self.sync_pool)} | {self.total_ack} acked / {self.recv_count} total ] Current {self.send_count} sent, {self.ack_advantages} ack. Speed {(self.recv_count-last_count)/interval:.2f}/s.")
    
    def _sync_node_queue(self):
        """节点队列同步线程"""
        while True:
            time.sleep(3)
            with self.sync_lock:
                self.sync_sender.send_multipart([
                    b"SYNC_NODE_QUEUE_LENGTHS",
                    pickle.dumps(self.node_queue_lengths)
                ])
            logger.debug(f"Sync node queue lengths {len(self.node_queue_lengths)}")
    
    def start(self):
        """启动同步管理器"""
        # 启动监控线程
        monitor_thd = threading.Thread(target=self._monitor, daemon=True)
        monitor_thd.start()
        
        # 启动节点队列同步线程
        sync_node_queue_thd = threading.Thread(target=self._sync_node_queue, daemon=True)
        sync_node_queue_thd.start()
        
        # 主循环
        self._run_main_loop()

    def _handle_ack(self, ack_count: int):
        """处理确认消息 (int)"""
        self.ack_advantages += ack_count
        self.total_ack += ack_count
        self.task_collect.send_string(f"Recived")
        
        if self.ack_advantages >= self.num_to_sync:
            # 同步所有设备进行更新
            with self.sync_lock:
                self.sync_sender.send_multipart([
                    b"SYNC_FOR_UPDATE",
                    pickle.dumps(self.sync_steps)
                ])
            self.ack_advantages -= self.num_to_sync
            self.send_count -= self.num_to_sync # 假设send_count在发送时增加

    def _handle_sync_request(self, request: SyncAdvantagesRequest):
        """处理同步优势请求 (SyncAdvantagesRequest)"""
        assert request.gid in self.sync_pool, f"Group {request.gid} not in SyncPool"
        self.task_collect.send_pyobj(self.sync_pool[request.gid])
        self.sync_count[request.gid] += 1
        if self.sync_count[request.gid] == self.num_nodes:
            # 所有节点已确认此任务
            del self.sync_count[request.gid]
            del self.sync_pool[request.gid]
            logger.debug(f"Sync advantages for {request.gid} completed")

    def _handle_queue_update(self, queue_lengths: dict):
        """处理节点队列长度更新 (dict)"""
        self.node_queue_lengths.update(queue_lengths)
        self.task_collect.send_string("Recived node queue lengths")

    def _handle_task_status(self, task_status: TaskStatus):
        """处理任务状态 (TaskStatus)"""
        self.task_collect.send_string(f"Recived completion {task_status.completion_id}")
        self.recv_count += 1
        self.task_collection[task_status.task_id].append(task_status)
        
        # 如果任务完成，计算并同步优势值
        if len(self.task_collection[task_status.task_id]) == self.num_generations:
            completed_task_group = self.task_collection.pop(task_status.task_id)
            
            scores = np.array([ts.score for ts in completed_task_group])
            advantages = (scores - scores.mean()) / (scores.std() + 1e-2)
            advantages = advantages.tolist()
            
            # 更新任务状态中的优势值
            for status, adv in zip(completed_task_group, advantages):
                status.advantage = adv
            
            # 将处理后的任务组放入同步池
            self.sync_pool[self.current_gid] = completed_task_group
            
            # 发送同步优势信号
            with self.sync_lock:
                self.sync_sender.send_multipart([
                    b"SYNC_ADVANTAGES",
                    pickle.dumps(self.current_gid)
                ])
            
            gid_to_sync = self.current_gid
            self.current_gid += 1
            
            # 更新发送计数器
            self.send_count += self.num_generations * self.tp_size
            if scores.mean() > DEFAULT_THRESHOLD or len(set(advantages)) == 1:
                # 这些任务将被丢弃，减少发送计数
                self.send_count -= self.num_generations * self.tp_size
                logger.debug(f"Group {gid_to_sync} tasks likely dropped due to high score or uniform advantage.")
            else:
                logger.debug(f"Group {gid_to_sync} advantages calculated and ready for sync.")


    def _run_main_loop(self):
        """主事件循环，接收消息并分发给相应的处理函数"""
        while True:
            # 接收消息
            message: Any = self.task_collect.recv_pyobj()
            
            # 根据消息类型调用不同的处理函数
            if isinstance(message, int):
                self._handle_ack(message)
            elif isinstance(message, SyncAdvantagesRequest):
                self._handle_sync_request(message)
            elif isinstance(message, dict):
                self._handle_queue_update(message)
            elif isinstance(message, TaskStatus):
                self._handle_task_status(message)
            elif isinstance(message, str):
                # 处理字符串消息（如果需要）
                logger.warning(f"Received unexpected string message: {message}")
                # 可能需要发送一个响应，即使是错误响应
                try:
                    self.task_collect.send_string("Error: Unexpected string message")
                except zmq.ZMQError as e:
                    logger.error(f"Error sending reply for string message: {e}")
                # raise NotImplementedError(f"Received string: {message}") # 或者记录错误并继续
            else:
                # 处理未知类型的消息
                logger.error(f"Received unknown message type: {type(message)}")
                try:
                    self.task_collect.send_string("Error: Unknown message type")
                except zmq.ZMQError as e:
                    logger.error(f"Error sending reply for unknown message type: {e}")

class LocalBalanceManager:
    """平衡本地机器创建的数据和任务，并与全局同步。"""
    
    def __init__(
        self,
        local_collect_address: str,
        local_provider_address: str,
        local_steal_port: int,
        global_sync_address: str,
        global_result_collect_address: str,
        global_data_dispatch_address: str,
        chunk_size: int,
        mt_max_beam_width: int,
        max_cache_size: int,
        processing_class_name_or_path: str,
        max_prompt_length: int,
        steal_threshold: int = 16,
        tp_size: int = 1,
        timeout: int = 3600
    ):
        """初始化本地平衡管理器。
        
        Args:
            local_collect_address: 本地收集地址
            local_provider_address: 本地提供者地址
            local_steal_port: 本地窃取端口
            global_sync_address: 全局同步地址
            global_result_collect_address: 全局结果收集地址
            global_data_dispatch_address: 全局数据分发地址
            chunk_size: 块大小
            mt_max_beam_width: 最大束宽度
            max_cache_size: 最大缓存大小
            processing_class_name_or_path: 处理器类名或路径
            max_prompt_length: 最大提示长度
            steal_threshold: 窃取阈值
            tp_size: 张量并行大小
            timeout: 超时时间
        """
        self.local_collect_address = local_collect_address
        self.local_provider_address = local_provider_address
        self.local_steal_port = local_steal_port
        self.global_sync_address = global_sync_address
        self.global_result_collect_address = global_result_collect_address
        self.global_data_dispatch_address = global_data_dispatch_address
        self.chunk_size = chunk_size
        self.mt_max_beam_width = mt_max_beam_width
        self.max_cache_size = max_cache_size
        self.processing_class_name_or_path = processing_class_name_or_path
        self.max_prompt_length = max_prompt_length
        self.steal_threshold = steal_threshold
        self.tp_size = tp_size
        self.timeout = timeout
        
        # 初始化ZMQ上下文
        self.zmqctx = zmq.Context(16)
        
        # 初始化所有ZMQ套接字
        self._init_sockets()
        
        # 初始化缓存和队列
        self.cached_tasks = {}
        self.global_ready_queue_length = {self.steal_addr: 0}
        self.valid_tasks = queue.Queue(max_cache_size // 4 * 3)
        self.ready_queue = queue.Queue(max_cache_size // 4)
        self.local_gid = 0
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(processing_class_name_or_path, trust_remote_code=True)
    
    def _init_sockets(self):
        """初始化所有ZMQ套接字和网络连接"""
        # 队列同步器
        self.queue_syncer = self.zmqctx.socket(zmq.REQ)
        self.queue_syncer.connect(self.global_result_collect_address)
        
        # 获取本机IP地址
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        parsed = urlparse(self.global_result_collect_address)
        try:
            s.connect((parsed.hostname, parsed.port))
            node_ip = s.getsockname()[0]
        finally:
            s.close()
        
        # 设置窃取地址和套接字
        self.steal_addr = f"tcp://{node_ip}:{self.local_steal_port}"
        self.steal_recv = self.zmqctx.socket(zmq.REP)
        self.steal_recv.bind(self.steal_addr)
        logger.info(f"Listen for stealing at {self.steal_addr}")
        
        # 同步信号订阅者
        self.sync_signal = self.zmqctx.socket(zmq.SUB)
        self.sync_signal.setsockopt(zmq.SUBSCRIBE, b"SYNC_ADVANTAGES")
        self._set_tcp_keepalive(self.sync_signal)
        self.sync_signal.connect(self.global_sync_address)
        
        # 队列同步订阅者
        self.sync_queue = self.zmqctx.socket(zmq.SUB)
        self.sync_queue.setsockopt(zmq.SUBSCRIBE, b"SYNC_NODE_QUEUE_LENGTHS")
        self._set_tcp_keepalive(self.sync_queue)
        self.sync_queue.connect(self.global_sync_address)
        
        # 结果发送者
        self.result_sender = self.zmqctx.socket(zmq.REQ)
        self.result_sender.connect(self.global_result_collect_address)
        
        # 任务分发发送者
        self.task_dispatch_sender = self.zmqctx.socket(zmq.REQ)
        self.task_dispatch_sender.connect(self.global_data_dispatch_address)
        
        # 优势同步器
        self.advantage_syncer = self.zmqctx.socket(zmq.REQ)
        self.advantage_syncer.connect(self.global_result_collect_address)
        
        # 平衡收集器
        self.balance_collect = self.zmqctx.socket(zmq.REP)
        self.balance_collect.bind(self.local_collect_address)
        
        # 平衡提供者
        self.balance_provider = self.zmqctx.socket(zmq.REP)
        self.balance_provider.bind(self.local_provider_address)
    
    def _set_tcp_keepalive(self, socket):
        """设置TCP保持连接选项"""
        socket.setsockopt(zmq.TCP_KEEPALIVE, 1)
        socket.setsockopt(zmq.TCP_KEEPALIVE_IDLE, 60)
        socket.setsockopt(zmq.TCP_KEEPALIVE_CNT, 5)
        socket.setsockopt(zmq.TCP_KEEPALIVE_INTVL, 10)
    
    def reprocess(self):
        """处理任务并生成处理后的数据块"""
        while True:
            collected = [self.valid_tasks.get().data for _ in range(self.chunk_size)]
            chunk_data = _process_inputs(collected, self.processor, self.max_prompt_length)
            self.ready_queue.put(chunk_data)
    
    def provider(self):
        """为工作进程提供数据"""
        cached_group_data = defaultdict(dict)
        visited_counts = defaultdict(dict)
        
        while True:
            tp_gid, rank, recv_idx = self.balance_provider.recv_pyobj()
            if cached_group_data[tp_gid].get(recv_idx, None) is None:
                # 该组的新数据
                chunk_data = self.ready_queue.get()
                cached_group_data[tp_gid][recv_idx] = pickle.dumps(chunk_data)
                visited_counts[tp_gid][recv_idx] = 0
            
            # 重用数据
            chunk_data = cached_group_data[tp_gid][recv_idx]
            visited_counts[tp_gid][recv_idx] += 1
            
            if visited_counts[tp_gid][recv_idx] == self.tp_size:
                logger.debug(f"Remove Group {tp_gid}, data id: {recv_idx}")
                # 所有设备都已接收此数据
                del cached_group_data[tp_gid][recv_idx]
                del visited_counts[tp_gid][recv_idx]
            
            self.balance_provider.send(chunk_data)
    
    def reporter(self):
        """报告队列状态"""
        while True:
            time.sleep(3)
            self.queue_syncer.send_pyobj({self.steal_addr: self.ready_queue.qsize()})
            self.queue_syncer.recv()
    
    def serve_stealing(self):
        """处理其他节点的任务窃取请求"""
        while True:
            nums_to_steal = self.steal_recv.recv_pyobj()
            mean_queue_length = sum(self.global_ready_queue_length.values()) / len(self.global_ready_queue_length)
            
            # 确定要发送多少
            nums_to_offer = max(int(min((self.ready_queue.qsize() - mean_queue_length) / 2, nums_to_steal)), 0)
            
            if nums_to_offer <= 0:
                self.steal_recv.send_pyobj(None)
            else:
                chunk_datas = []
                try:
                    for _ in range(nums_to_offer):
                        chunk_datas.append(self.ready_queue.get_nowait())
                except:
                    pass
                self.steal_recv.send_pyobj(chunk_datas)
    
    def work_stealing(self):
        """从其他节点窃取任务"""
        while True:
            parts = self.sync_queue.recv_multipart()
            if len(parts) != 2:
                logger.error(f"Invalid message parts: {len(parts)}, parts: {parts}")
                continue
            
            _, d = parts
            self.global_ready_queue_length.update(pickle.loads(d))
            if len(self.global_ready_queue_length) == 1:
                continue
            
            current_q = self.ready_queue.qsize()
            mean_queue_length = sum(self.global_ready_queue_length.values()) / len(self.global_ready_queue_length)
            
            starving = mean_queue_length - current_q
            if starving > self.steal_threshold:
                nums_to_steal = int(starving / 2)
            else:
                nums_to_steal = 0
            
            if nums_to_steal > 0:
                # 选择目标
                items = [(addr, qlen) for addr, qlen in self.global_ready_queue_length.items() 
                         if addr != self.steal_addr and qlen > mean_queue_length]
                if not items:
                    continue
                
                addr, remote_q = random.choice(items)
                # 随机选择
                nums_to_steal = min(nums_to_steal, int((remote_q - mean_queue_length) / 2))
                
                # 进行窃取
                with self.zmqctx.socket(zmq.REQ) as steal_req:
                    steal_req.connect(addr)
                    steal_req.send_pyobj(nums_to_steal)
                    stealed = steal_req.recv_pyobj()
                
                if stealed:
                    for chunk_data in stealed:
                        self.ready_queue.put(chunk_data)
    
    def sync_handler(self):
        """处理同步信号并更新任务状态"""
        while True:
            parts = self.sync_signal.recv_multipart()
            if len(parts) != 2:
                logger.error(f"Invalid message parts: {len(parts)}, parts: {parts}")
                continue
            
            topic, d = parts
            latest_gid = pickle.loads(d)
            
            while self.local_gid <= latest_gid:
                logger.debug(f"Sync advantages for group {self.local_gid}")
                self.advantage_syncer.send_pyobj(SyncAdvantagesRequest(self.local_gid))
                task_status = self.advantage_syncer.recv_pyobj()
                task_status.sort(key=lambda x: x.advantage, reverse=True)
                
                next_new_tasks = []
                valid_next_task_completions = []
                scores = []
                
                for status in task_status:
                    scores.append(status.score)
                    if (status.score >= float(os.environ.get("MULTITURN_SAMPLE_THRESHOLD", DEFAULT_THRESHOLD)) and 
                        status.advantage >= task_status[self.mt_max_beam_width % len(task_status)].advantage):
                        # 考虑将下一轮任务添加到任务队列
                        valid_next_task_completions.append(status.completion_id)
                
                # 检查是否可能有进一步的任务
                for status in task_status:
                    if status.completion_id in self.cached_tasks:
                        d = self.cached_tasks[status.completion_id]
                        if (d.data.get("next_id", None) is not None and 
                            d.status.completion_id in valid_next_task_completions):
                            # 考虑将下一轮任务添加到任务队列
                            next_new_tasks.append(d)
                
                if next_new_tasks:
                    # 发送到全局任务分发循环
                    data = []
                    for item in next_new_tasks:
                        data.append({
                            "id": item.data["id"],
                            "gid": self.local_gid,
                            "next_id": item.data["next_id"],
                            "completion_id": item.status.completion_id,
                            "completion": item.data["completion"],
                        })
                    
                    # 计算发送数据的正确长度
                    valid_completions_counts = len(valid_next_task_completions)
                    ratio = len(task_status) // valid_completions_counts
                    data = data * ratio
                    
                    if (rem := len(task_status) % valid_completions_counts) != 0:
                        # 仅对一个节点附加提醒
                        # 选择包含最小字典顺序UUID的节点并添加提醒
                        small_completion_id = sorted(valid_next_task_completions)[0]
                        # 验证当前数据批次中是否包含最早的任务
                        contains_smallest = False
                        for item in data:
                            if item["completion_id"] == small_completion_id:
                                contains_smallest = True
                                break
                        
                        if contains_smallest:
                            data += data[:1] * rem
                    
                    self.task_dispatch_sender.send_pyobj(data)
                    self.task_dispatch_sender.recv()
                
                scores = np.array(scores)
                if scores.mean() > DEFAULT_THRESHOLD or len(set(map(lambda x: x.advantage, task_status))) == 1:
                    # 检查缓存的任务是否可以更新并发回进行反向传播
                    # 我们应该直接丢弃任务
                    drop = []
                    for status in task_status:
                        if status.completion_id in self.cached_tasks:
                            drop.append(self.cached_tasks.pop(status.completion_id))
                    
                    if drop:
                        logger.debug(f"Drop {len(drop)} tasks in Group {self.local_gid}, "
                                     f"Cache size: {len(self.cached_tasks) + len(drop)} -> {len(self.cached_tasks)}")
                    del drop
                else:
                    # 有不同的优势
                    pre_len = len(self.cached_tasks)
                    for status in task_status:
                        if status.completion_id in self.cached_tasks:
                            self.cached_tasks[status.completion_id].status.advantage = status.advantage
                            d = self.cached_tasks.pop(status.completion_id)
                            d.data["advantage"] = d.status.advantage
                            self.valid_tasks.put(d)
                            
                            if d.status.completion_id == task_status[0].completion_id:
                                # 最佳任务
                                logger.info("Best Completion (%.2f) : %s", d.status.score, d.data["completion"])
                    
                    logger.debug(f"Sync {len(task_status)} tasks in Group {self.local_gid}, "
                                 f"Cache size: {pre_len} -> {len(self.cached_tasks)}")
                
                self.local_gid += 1
    
    def monitor(self):
        """检查缓存中是否有超时的任务"""
        interval = int(os.environ.get("LOCAL_MONITOR_INTERVAL", "300"))
        while True:
            time.sleep(interval)
            oldest_task = None
            
            for k, v in list(self.cached_tasks.items()):
                if oldest_task is None or v.status.created_time < oldest_task.status.created_time:
                    oldest_task = v
                
                if (datetime.datetime.now() - v.status.created_time).total_seconds() > self.timeout:
                    logger.warning(f"Task {v.status.task_id}, completion {k} is out of time.")
                    # del self.cached_tasks[k]
            
            if oldest_task is not None:
                logger.info(f"[ Local GID: {self.local_gid} | Cached: {len(self.cached_tasks)}, "
                            f"Reprocessing: {self.valid_tasks.qsize()} | Queued: {self.ready_queue.qsize()} ] "
                            f"The oldest task id {oldest_task.status.task_id}, "
                            f"create time: {oldest_task.status.created_time}")
    
    def start(self):
        """启动所有线程并运行主循环"""
        # 启动所有线程
        threads = [
            threading.Thread(target=self.reprocess, daemon=True),
            threading.Thread(target=self.provider, daemon=True),
            threading.Thread(target=self.reporter, daemon=True),
            threading.Thread(target=self.serve_stealing, daemon=True),
            threading.Thread(target=self.work_stealing, daemon=True),
            threading.Thread(target=self.sync_handler, daemon=True),
            threading.Thread(target=self.monitor, daemon=True)
        ]
        
        for thread in threads:
            thread.start()
        
        # 主循环
        while True:
            tac: TaskAndContent = self.balance_collect.recv_pyobj()
            self.cached_tasks[tac.status.completion_id] = tac
            self.balance_collect.send_string("Received")
            
            # 发送任务状态到全局
            self.result_sender.send_pyobj(tac.status)
            self.result_sender.recv_string()
            
            if len(self.cached_tasks) >= self.max_cache_size:
                logger.warning(f"Too many cached tasks. [ Local GID: {self.local_gid} | "
                               f"Cached: {len(self.cached_tasks)}, Reprocessing: {self.valid_tasks.qsize()} | "
                               f"Queued: {self.ready_queue.qsize()} ]")
                time.sleep(3)  # 缓存过多
