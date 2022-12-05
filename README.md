# Sequence-to-Function-Learning

This repo contains work on sequence-to-function learning based on language models.

The model was trained on 1000 PafA single amino acid mutants and its prediction performance was also shown. 


## Prediction Pipeline
<p align="center">
  <img width="900"  src="https://user-images.githubusercontent.com/47986787/205684697-7675f4fc-f821-4218-aede-8979aaac8789.png">
</p>


## Prediction Performance
<p align="center">
  <img width="600"  src="https://user-images.githubusercontent.com/47986787/205684314-7029ee8f-f1cb-4375-9bd2-b881750b015c.png">
</p>

### Make a $y$ vs. $\hat{x}$ plot.
```
from ZX01_PLOT import *
reg_scatter_distn_plot(y_pred_valid,
                        y_real_valid,
                        fig_size        =  (10,8),
                        marker_size     =  35,
                        fit_line_color  =  "brown",
                        distn_color_1   =  "gold",
                        distn_color_2   =  "lightpink",
                        # title         =  "Predictions vs. Actual Values\n R = " + \
                        #                         str(round(r_value,3)) + \
                        #                         ", Epoch: " + str(epoch+1) ,
                        title           =  "",
                        plot_title      =  "R = " + str(round(r_value,3)) + \
                                                  "\nEpoch: " + str(epoch+1) ,
                        x_label         =  "Actual Values",
                        y_label         =  "Predictions",
                        cmap            =  None,
                        cbaxes          =  (0.425, 0.055, 0.525, 0.015),
                        font_size       =  18,
                        result_folder   =  results_sub_folder,
                        file_name       =  output_file_header + "_VA_" + "epoch_" + str(epoch+1),
                        ) #For checking predictions fittings.


```


