from rot_data.utils.logging import setup_logger


def main():
    # Initialize logger with Rich handler for better compatibility
    setup_logger(level="INFO", use_rich=True)
    print("Hello from rot-data!")


if __name__ == "__main__":
    main()
