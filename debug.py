import subprocess
with open("requirements.txt", "w") as f:
    subprocess.run(["python", "-m", "pip", "freeze"], stdout=f)
print("✅ requirements.txt created!")
