#Import Necessary packages
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

def plot_history1(history, metric1, metric2):
    fig, (ax_loss, ax_score) = plt.subplots(1, 2, figsize=(15,5))
    ax_loss.plot(history.epoch, history.history[metric1], label=metric1)
    #ax_loss.plot(history.epoch, history.history["val_loss"], label="Validation loss")
    ax_loss.legend()
    ax_score.plot(history.epoch, history.history[metric2], label=metric2)
    #ax_score.plot(history.epoch, history.history["val_my_iou_metric"], label="Validation score")
    ax_score.legend()
