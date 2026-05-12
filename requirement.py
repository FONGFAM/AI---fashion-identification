import subprocess
import sys

def install_packages():
    """Cài đặt các thư viện cần thiết cho dự án Nhận diện thời trang."""
    libraries = [
        "flask",
        "pillow",
        "numpy",
        "tensorflow"
    ]
    
    print("Dang tien hanh cai dat cac thu vien can thiet...")
    
    for lib in libraries:
        print(f"Dang cai dat {lib}...")
        try:
            # Chạy lệnh pip install thông qua subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])
            print(f"Da cai dat thanh cong {lib}!\n")
        except subprocess.CalledProcessError as e:
            print(f"Loi khi cai dat {lib}. Chi tiet: {e}\n")
        except Exception as e:
            print(f"Loi khong xac dinh khi cai {lib}: {e}\n")
            
    print("Hoan tat qua trinh tai thu vien!")

if __name__ == '__main__':
    install_packages()
