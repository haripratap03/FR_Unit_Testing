import os
import subprocess

folder = r"D:\1pharmacy\FR Analysis\Unit Testing\FR_Thor\Unit_Test_Data\Face_Detection\scripts"

for file in os.listdir(folder):
    if file.endswith(".py"):
        full_path = os.path.join(folder, file)
        print(f"Running {file}...")
        subprocess.run(["python", full_path])