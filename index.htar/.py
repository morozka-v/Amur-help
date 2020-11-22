from window import *


MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
	early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

	model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanSquaredError()])

	history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
	return history

def date_to_nth_day(date):
    date = pd.to_datetime(date)
    new_year_day = pd.Timestamp(year=date.year, month=1, day=1)
    return (date - new_year_day).days + 1

def nan_helper(y):
    return np.isnan(y), lambda z: z.nonzero()[0]

def main_func():
	path_to_data = 'C:/Users/Dino_bugaini/Desktop'
	daily = pd.read_pickle(path_to_data + '/daily.pkl')
	print(daily.head())
	stations = []
	for i in daily['station_id']:
	    if i not in stations:
	        stations.append(i)

	for i in stations[:1]:
	    plot_cols=['stage_avg', 'temp']
	    station = daily[daily.station_id == i]
	    plot_features = station[plot_cols]
	    plot_features.index=station['date']
	    _ = plot_features.plot(subplots=True)
	    
	    plot_features = station[plot_cols][:365]
	    plot_features.index = station['date'][:365]
	    _ = plot_features.plot(subplots=True)
	    plt.show()

	for i in stations:
	    station = daily[daily.station_id == i]
	    fft = tf.signal.rfft(station['stage_avg'])
	    f_per_dataset = np.arange(0, len(fft))

	    n_samples_d = len(station['stage_avg'])
	    days_per_year = 365.254
	    years_per_dataset = n_samples_d/(days_per_year)
	    f_per_year = f_per_dataset/years_per_dataset
	    plt.step(f_per_year, np.abs(fft))
	    plt.xscale('log')
	    plt.ylim(0, 1700000)
	    plt.xlim([0.01, max(plt.xlim())])
	    plt.xticks([1,3], labels=['1/Year','1/122 days'])
	    _ = plt.xlabel('Frequency (log scale)')

	for i in stations:
	    station = daily[daily.station_id == i]
	    fft = tf.signal.rfft(station['temp'])
	    f_per_dataset = np.arange(0, len(fft))

	    n_samples_d = len(station['temp'])
	    days_per_year = 365.254
	    years_per_dataset = n_samples_d/(days_per_year)
	    f_per_year = f_per_dataset/years_per_dataset
	    plt.step(f_per_year, np.abs(fft))
	    plt.xscale('log')
	    plt.ylim(0, 20000)
	    plt.xlim([0.1, max(plt.xlim())])
	    plt.xticks([1], labels=['1/Year'])
	    _ = plt.xlabel('Frequency (log scale)')

	day_of_year = np.arange(1,367)
	print(len(day_of_year))
	sin = np.sin(2*np.pi*day_of_year/len(day_of_year))
	cos = np.cos(2*np.pi*day_of_year/len(day_of_year))
	sin_p = []
	cos_p = []
	for i in daily.date:
	    sin_p.append(sin[date_to_nth_day(i)-1])
	    cos_p.append(cos[date_to_nth_day(i)-1])
	daily['Year sin'] = sin_p
	daily['Year cos'] = cos_p
	for i in stations[:1]:
	    plot_cols=['Year sin', 'Year cos']
	    station = daily[daily.station_id == i]
	    plot_features = station[plot_cols]
	    plot_features.index=station['date']
	    _ = plot_features.plot(subplots=True)
	    
	    plot_features = station[plot_cols][:365]
	    plot_features.index = station['date'][:365]
	    _ = plot_features.plot(subplots=True)

	daily = daily.drop(columns = ['date','stage_min','stage_max','water_code'])
	num_features = 4
	y = daily.stage_avg.to_numpy()
	nans, x= nan_helper(y)
	y[nans]= np.interp(x(nans), x(~nans), y[~nans])
	daily.stage_avg = y
	daily = daily.fillna({ 'temp' : 0})
	OUT_STEPS = 10
	multi_window = WindowGenerator(input_width=90,
	                               label_width=OUT_STEPS,
	                               shift=OUT_STEPS, station=5001, daily=daily, label_columns=['stage_avg'])

	multi_window.plot()
	multi_window
	num_features = 4
	multi_window.plot_normal_data()
	multi_linear_model = tf.keras.Sequential([
	    # Take the last time-step.
	    # Shape [batch, time, features] => [batch, 1, features]
	    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
	    # Shape => [batch, 1, out_steps*features]
	    tf.keras.layers.Dense(OUT_STEPS*num_features,
	                          kernel_initializer=tf.initializers.zeros),
	    # Shape => [batch, out_steps, features]
	    tf.keras.layers.Reshape([OUT_STEPS, num_features])
	])
	
	multi_val_performance = {}
	multi_performance = {}
	history = compile_and_fit(multi_linear_model, multi_window)


	multi_val_performance['Dense'] = multi_linear_model.evaluate(multi_window.val)
	multi_performance['Dense'] = multi_linear_model.evaluate(multi_window.test, verbose=0)
	multi_window.plot(multi_linear_model)


main_func()
