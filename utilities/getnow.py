from datetime import datetime

def get_current_time_formatted():
    # Get the current date and time
    now = datetime.now()

    # Format the date and time as a string
    time_string = now.strftime("%Y-%m-%d %H:%M:%S")

    return f"{time_string} is the current date and time"

# Example usage
# current_time = get_current_time_formatted()
# print(current_time)
