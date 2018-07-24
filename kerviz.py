import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec

# prevents overflow when displaying a lot of data points
mpl.rcParams['agg.path.chunksize'] = 100000

# do some set up and return a closure around the redraw callback for Keras 
def get_redraw(X_in, y_in, model, batch_size, epochs, **kwargs):

    ## PROCESS COMMAND LINE ARGUMENTS
    
    # plot dimensions
    left = kwargs.get('left', X_in.min())
    right = kwargs.get('right', X_in.max())
    bottom = kwargs.get('bottom', y_in.min())
    top = kwargs.get('top', y_in.max())

    # how much data to use in graph
    
    # ... scatter plot sparsity (0 = no scatter plot)
    #     scatter is only drawn once, but it can be a lot of data, both computationally and visually
    scatter_sparsity = kwargs.get('scatter_sparsity', 5)

    # ... graph sparsity
    #     keeping the graph sparse improves performance
    graph_sparsity = kwargs.get('graph_sparsity', 1000)

    # whether to display loss
    show_loss = kwargs.get('show_loss', True)
    
    # .. and level of smoothing to apply to loss, if so (needs to be an odd number)
    loss_smoothing = kwargs.get('loss_smoothing', 101)
    
    # how frequently (in batches) to update the graph
    frequency = kwargs.get('frequency', 10)
    if callable(frequency):
        # if a function is provided, it will be called every batch and asked for a True/False response
        frequency_mode = 'function'
    elif np.isscalar(frequency):
        # if a number is provided, updates will be done every [frequency] batches
        frequency_mode = 'scalar'
    else:
        # for array-like setting, update when frequency[batch number] is True
        frequency_mode = 'array'
        
    # figure size
    figure_size = kwargs.get('figure_size', (15, 10))
    
    # text labels
    title = kwargs.get('title', None)
    x_label = kwargs.get('x_label', None)
    y_label = kwargs.get('y_label', None)
    
    # tick formatters
    x_tick_formatter = kwargs.get('x_tick_formatter', None)
    y_tick_formatter = kwargs.get('y_tick_formatter', None)
    
    # loss scale (depends on loss function)
    loss_scale = kwargs.get('loss_scale', 1.0)
    
    # display legend
    show_legend = kwargs.get('show_legend', True)
    
    # write to screen or file?
    display_mode = kwargs.get('display_mode', 'screen')
    filepath = kwargs.get('filepath', 'images/batch')
    
    
    ## PREP DATA FOR QUICKER DISPLAY
    
    # parallel sort feature and label arrays for plotting
    ix = X_in.argsort(axis=0)[:,0]

    # ... reducing number of points used in graph
    ix = ix[::graph_sparsity]
    
    X = X_in[ix]
    y = y_in[ix]
    
    # keep track of total number of batches seens
    tot_batches = 0
    batches_per_epoch = np.ceil(len(X_in) / batch_size)
    
    # scale for loss plot = total number of batches that will be run
    max_batches = epochs * batches_per_epoch
    
    ## DRAW BACKGROUND COMPONENTS
    
    # set the figure size and layout
    fig = plt.figure(figsize=figure_size)
    grd = gridspec.GridSpec(ncols=3, nrows=2)
    
    # graphs for data/loss, loss over time, and model parameters
    ax_main = fig.add_subplot(grd[:2, :2])
    ax_loss = fig.add_subplot(grd[:1, 2:])
    ax_params = fig.add_subplot(grd[1:, 2:])

    # data boundaries on main graph
    ax_main.set_xlim(left, right)
    ax_main.set_ylim(bottom, top)
    
    # titles and labels on main graph
    if title:
        ax_main.set_title(title, size=14, fontweight='bold', y=1.03)
    
    if x_label:
        ax_main.set_xlabel(x_label, size=12, fontweight='bold')
        
    if y_label:
        ax_main.set_ylabel(y_label, size=12, fontweight='bold')
                 
    # tick formatting on main graph
    if x_tick_formatter:
        ax_main.xaxis.set_major_formatter(x_tick_formatter)
        
    if y_tick_formatter:
        ax_main.yaxis.set_major_formatter(y_tick_formatter)
        
    # draw a scatter plot of the training data on main graph
    if scatter_sparsity > 0:
        ax_main.scatter(X_in[::scatter_sparsity], y_in[::scatter_sparsity], marker='.', c='silver', s=1, alpha=0.5, zorder=10)

    # set titles and labels on loss plots
    ax_loss.set_title("Total Loss", size=11, fontweight='bold', y=0.9)
    ax_loss.set_xlabel("Batch", size=9, fontweight='bold')
    ax_loss.set_ylabel("Loss", size=9, fontweight='bold')

    ax_params.set_title("Batch Loss", size=11, fontweight='bold', y=0.9)
    ax_params.set_xlabel("Batch", size=9, fontweight='bold')
    ax_params.set_ylabel("Loss", size=9, fontweight='bold')

    # set scale of loss plots
    # x axes are logarithimic because progress slows over course of training
    ax_loss.set_xscale('log', nonposx='clip')
    ax_loss.set_xlim(1, max_batches)
    ax_loss.set_ylim(0, loss_scale)        

    ax_params.set_xscale('log', nonposx='clip')
    ax_params.set_xlim(1, max_batches)
    ax_params.set_ylim(0, loss_scale)        

    if display_mode == 'file':
        plt.savefig("%s-%05d.png" %(filepath, 0))
    
    # declare components that will be retained between calls
    first_pass = True
    y_pred_line = None
    loss_line_u = None
    loss_line_d = None
    fill_between = None
    
    # RETURN A CALLBACK FUNCTION usable by keras with closure around fixed arguments
    def redraw(batch, logs):
        
        # let Python know that outside scope variables will be used
        nonlocal first_pass, y_pred_line, loss_line_u, loss_line_d, fill_between, tot_batches

        # keep track of total number of batches seens
        tot_batches += 1
                
        # update graph at the requested frequency
        
        if frequency_mode == 'scalar':
            if tot_batches % frequency != 0:
                return
        elif frequency_mode == 'array':
            if not frequency[tot_batches]:
                return    
        
        if frequency_mode == 'function':
            if not frequency(model, X, y, tot_batches):
                return
        
        # run the model in its current state of training to get the prediction so far
        y_pred = model.predict(X).reshape(-1)
        
    
        if show_loss:
            
            # compute the loss relative to each training label
            loss = np.square(y - y_pred.reshape(-1))

            # smooth loss with a moving average 
            if loss_smoothing > 1:
                loss = np.convolve(loss, np.ones((loss_smoothing,))/loss_smoothing, mode='same')

        # first time through, draw the dynamic portions
        if first_pass:

            # draw the current prediction of the model
            y_pred_line = ax_main.plot(X, y_pred, '-', color='steelblue', lw=4, label='model', zorder=15)[0]

            if show_loss:

                # draw the lossor around the prediction
                loss_line_u = ax_main.plot(X, y_pred + loss , '-', alpha=0.6, lw=0.5, color='steelblue', label='loss', zorder=3)[0]
                loss_line_d = ax_main.plot(X, y_pred - loss, '-', alpha=0.6, lw=0.5, color='steelblue', zorder=3)[0]

            if display_mode == 'screen':
                plt.show()

            first_pass = False

        # on subsequent calls, update the dynamic portions
        else:

            # draw the current prediction of the model
            y_pred_line.set_ydata(y_pred)

            # update the loss around the prediction
            if show_loss:
                loss_line_u.set_ydata(y_pred + loss)
                loss_line_d.set_ydata(y_pred - loss)

        if show_loss:
            
            # shade in the area between the loss lines
            if fill_between:
                fill_between.remove()

            fill_between = ax_main.fill_between(X.reshape(-1), y_pred + loss, y_pred - loss, color='steelblue', alpha=0.2, zorder=0)

        # add points to loss graphs
        tot_loss = loss.sum() / len(y)
        ax_loss.scatter([tot_batches], [tot_loss], s=5, c='steelblue')            
        ax_params.scatter([tot_batches], [logs['loss']], s=5, c='steelblue')            
            
        if show_legend:
            ax_main.legend().set_zorder(20)        

        if display_mode == 'screen':

            # push changes to screen
            fig.canvas.draw()
            fig.canvas.flush_events()

        elif display_mode == 'file':

            # save changes to image file        
            plt.savefig("%s-%05d.png" % (filepath, tot_batches))
    
    # return the closure around the callback that Keras will use
    return redraw
