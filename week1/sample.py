import time

class Timer:
    def __enter__(self):
        print("1. [__enter__] 开始计时...")
        self.start = time.time()
        return self  # 返回值赋给 as 后面的 t

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f"3. [__exit__] 计时结束，耗时: {self.end - self.start:.4f} 秒")
        if exc_type is not None:
            print(f"   ⚠️ 注意：代码块内发生了异常 -> {exc_type.__name__}: {exc_val}")
        # 返回 False（默认）表示如果有异常，继续往外抛出；如果返回 True 则会吞掉异常
        return False

# 实际使用我们自定义的上下文管理器：
with Timer() as t:
    print("2. [with 块内部] 正在执行耗时计算...")
    total = sum(i * i for i in range(1000000))