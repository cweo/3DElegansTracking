# https://www.machinecurve.com/index.php/2019/10/08/how-to-visualize-the-training-process-in-keras/

history = model.fit(imgs_train, imgs_mask_train, batch_size=1, epochs=50, verbose=1, shuffle=True, validation_split=0.10, callbacks=[model_checkpoint, csv_logger])

plt.plot(history.history['accuracy'], label='Accuracy (testing data)')
plt.plot(history.history['val_accuracy'], label='Accuracy (validation data)')
plt.title('Accuracy over time during training')
plt.ylabel('Accuracy value')
plt.xlabel('No. epoch')
plt.legend(loc="upper left")
plt.show()