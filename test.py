#a resusable function to print a picture and show predictions

def show_predict(img, predict = False, pred_model = None, shape = (28, 28), threshold = 0.05, show_img = True):
    
    """
    an image display function for dataset
    you can determine whether predictions are also printed from a given model
    
    > img: image to print
    > predict: whether predictions are calculated and printed
    > pred_model: keras model that will be used for predictions
    > shape: image shape, 28x28 by default
    > threshold: minimum probability with which predictions will be displayed
    > show_img: if False, does not show the image
    
    """
    if show_img:
        plt.imshow(img.reshape(shape),vmin=0., vmax=1.)
        plt.show()
    
    if predict:
        
        if pred_model == None:
            print("you did not provide a model!")
            
        else:
            prediction = pred_model.predict(img)[0]
            prediction = [(f'number {range(10)[idx]}', prediction[idx]) for idx in range(10) if prediction[idx] >= threshold]
            print(prediction)