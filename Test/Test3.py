from MyLib import timeChecking

def printI(num: int) -> None:
    for i in range(num):
        print(i)

print(timeChecking(printI, 1000000))  # Example usage with an argument







