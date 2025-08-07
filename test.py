from datasets import load_dataset

from krill.utils.memory_monitor import MemoryMonitor


monitor = MemoryMonitor()
monitor.start_monitoring()

monitor.report_current("before loading datasets")

dataset = load_dataset("blueapple8259/TinyFinewebEdu-ko", split="train", streaming=True)

monitor.report_current("after loading datasets")

print(dataset)

monitor.report_current("after processing dataset")

monitor.report_final()
