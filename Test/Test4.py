import numpy as np

# Dữ liệu đầu vào
X = np.arange(0, 20, 2)
y = 1 + 3*X**2

# Đặc trưng đa thức bậc 2
degree = 2

# Tạo mảng 2D chứa đặc trưng đa thức
X_poly = np.zeros((X.shape[0], degree + 1))
# print(X.shape[0])
# print(X_poly)

# Tạo các đặc trưng đa thức bằng cách lặp qua bậc
for i in range(degree + 1):
    X_poly[:, i] = X**i

# Tìm hệ số bằng cách sử dụng phép chuyển vị và nhân ma trận
XTX = X_poly.T @ X_poly
XTy = X_poly.T @ y
theta = np.linalg.solve(XTX, XTy)

# Dự đoán và đánh giá mô hình
y_pred = X_poly @ theta

# Đánh giá hiệu suất, ví dụ: MSE
mse = np.mean((y_pred - y)**2)

# Trực quan hóa kết quả
import matplotlib.pyplot as plt

plt.scatter(X, y, label='Data')
plt.plot(X, y_pred, label='Polynomial Regression', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

print(f'MSE: {mse}')
