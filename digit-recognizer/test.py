import random
import matplotlib.pyplot as plt

dice_sums = []

for i in range(1000):
    dice_1 = random.randint(1, 6)
    dice_2 = random.randint(1, 6)
    # dice_3 = random.randint(1, 6)
    dice_sum = dice_1 + dice_2
    dice_sums.append(dice_sum)
    # print(f"{i + 1:>2}. [{dice_1}]", f"[{dice_2}]", f"[{dice_3}]: Tổng của xúc xắc -> {dice_sum}")

plt.hist(dice_sums, bins=range(2, 14), edgecolor='black')
plt.xlabel('Sum of Dice')
plt.ylabel('Frequency')
plt.title('Histogram of Dice Sums')
plt.show()
