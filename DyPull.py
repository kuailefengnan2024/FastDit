import subprocess

def git_pull():
    command = "git pull"
    print(f"Running command: {command}")
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    
    # 使用 errors='ignore' 来处理解码错误
    print(stdout.decode(errors='ignore'))
    print(stderr.decode(errors='ignore'))

if __name__ == "__main__":
    git_pull()

