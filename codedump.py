
salaries.plot.scatter("yearID", "salary")

x_train, x_test, y_train, y_test = train_test_split(salaries.yearID, salaries.salary, test_size = 0.2)
regr = LinearRegression()
regr.fit(np.array(x_train).reshape(-1,1), y_train)
