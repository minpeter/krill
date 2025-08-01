import argparse

def main():
    parser = argparse.ArgumentParser(description="My new CLI tool.")
    parser.add_argument("name", help="Name to greet.")
    args = parser.parse_args()
    print(f"Hello, {args.name}!")

# 이 파일이 직접 실행될 때를 위한 코드 (선택 사항)
if __name__ == "__main__":
    main()