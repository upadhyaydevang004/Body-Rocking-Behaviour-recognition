import matplotlib.pyplot as plt

max_depth= [80, 90]
max_features=[2, 3]
n_estimators= [100, 200, 300]
acc_depth = [0.9012,0.8512]
acc_est = [0.8376,0.8643,0.9154]
acc_feat = [0.9205,0.8872]

plt.figure()
plt.title("Accuracy")
plt.xlabel('Estimators')
plt.ylabel('Accuracy')
plt.plot(n_estimators,acc_est, label='Accuracy')

plt.figure()

plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.title("Accuracy")

plt.plot(max_depth,acc_depth, label='Accuracy')
plt.show()

plt.figure()
plt.title("Accuracy")
plt.xlabel('Features')
plt.ylabel('Accuracy')
plt.plot(max_depth,acc_feat, label='Accuracy')
plt.show()
