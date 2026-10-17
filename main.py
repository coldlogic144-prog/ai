print("================================")
print("       ColdLogic AI")
print("  Handwritten Digit Recognizer")
print("================================")

print("Welcome! Your AI project has started.")
try:
    num = int(input("enter a number between (0-9):"))
    if type(num)==float:
        print("invalid input enter a integer")
    else:
        if num>9 or num<0:
            print("invalid input")
        else:
                
                print("You entered:", num)
except ValueError:
    print("Invalid input! Please enter a valid whole number.")

