import serial
import time

# Set up the serial connection (adjust the port and baudrate to match your Arduino settings)
  # Replace 'COM3' with your port

ser = serial.Serial('COM4', 9600)

I_Hx = 1

def send_command(command):
    ser.write(command.encode())
    time.sleep(0.1)  # Small delay to ensure command is sent

def activateI(I_Mx, I_My):
    direction = I_My/I_Mx
    I_Hy = direction * I_Hx
    send_command('{},{},{},{}'.format(I_Mx,I_Hx,I_My,I_Hy))    


def main():
    while True:
        command = input("Enter command (w/a/s/d to move, c to stop, q to quit): ").lower()
        if command == 'a':
            move_left()
        elif command == 'd':
            move_right()
        elif command == 'w':
            move_up()
        elif command == 's':
            move_down()
        elif command == 'c':
            stop_all()
        elif command == 'q':
            print("Quitting...")
            break
        else:
            print("Invalid command. Please enter w/a/s/d to move, c to stop, or q to quit.")

    ser.close()

if __name__ == "__main__":
    main()
