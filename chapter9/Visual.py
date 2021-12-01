# -*- coding: utf-8 -*-

from kt_package.Personal_module import Creat_gif,print_time    

print_time()
image_list = ['Experiment6/Experiment6_generated_plot_e'+str(i)+'.png' for i in range(100,480,10)]
gif_name = 'Experiment6_generated_plot.gif'
duration = 0.5
Creat_gif(image_list, gif_name,duration)
print()