# startup.py

from huggingface_hub import login

def main():
    print("Logging into Hugging Face Hub...")
    login()  # This will prompt for your Hugging Face token interactively
    print("Logged in successfully!")

if __name__ == "__main__":
    main()
