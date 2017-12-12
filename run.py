import pickle
import time
import numpy as np
from modules import UNetModel
from modules import ImageGenerator


def run_neural_net(img_width):
    trn_gen, val_gen, tst_gen = ImageGenerator.get_generators(img_width)
    model, callbacks = UNetModel.get_unet_model(img_width)
    num_epochs = 10
    print('Training initialized\n')
    start = time.time()
    history = model.fit_generator(trn_gen,
                                  steps_per_epoch = 2035,
                                  epochs = num_epochs,
                                  validation_data = val_gen,
                                  validation_steps = 252,
                                  callbacks = callbacks)
    stop = time.time()
    print('Training complete\nSaving model')
    model.save('model.h5')
    pickle.save(history.history, open('history.p', 'wb'))
    
    trn_acc = history.history.get('dice_coef')
    val_acc = history.history.get('val_dice_coef')
    tst_acc = [model.evaluate_generator(tst_gen, steps = 252)[1] for _ in range(num_epochs)]
    print('Training Time:', stop - start, 'seconds')
    print('Average Training Accuracy: ', np.mean(trn_acc))
    print('Average Validation Accuracy: ', np.mean(val_acc))
    print('Average Testing Accuracy: ', np.mean(tst_acc))
    
if __name__ == '__main__':
    img_width = -1
    while(True):
        print('Smaller images allows training to be perfomed faster with a slight loss in accuracy\n')
        choice = input('Select the image scale factor\n(a) 1\n(b) 1/2\n(c) 1/4\n(d) 1/8\n')
        choice = choice.strip()
        if(choice == ''):
            print('Please try again')
        elif(choice.isalpha()):
            if(choice == 'a'):
                img_width = 1280
                break
            elif(choice == 'b'):
                img_width = 1280 // 2
                break
            elif(choice == 'c'):
                img_width = 1280 // 4
                break
            elif(choice == 'd'):
                img_width = 1280 // 8
                break
            else:
                print('Please try again')
    print('The image width is', img_width)
    run_neural_net(img_width)