

import download_stock
import fourier_func



stock_input = input('ENTER stock TICKER and press ENTER >> ')
stock = stock_input.upper()

df = download_stock.download(stock)
print(df)


fourier_func.func_fourier_with_power_plot(df, stock, frequency_threshold=0.1, sampling_rate_hz=1.0, plot=True,
                                         plot_power=False, plot_phase=False, plot_PowerLaw=False)