
# image generator
def gen_and_plot_random(encoder,decoder,device,row,col,noise_factor=0.3):
    plt.figure(figsize=(10,4.5))
    
    for r in range(row):
      for c in range(col):
        ax = plt.subplot(row,col,r*col + c+1)
        random_encoded_image = torch.randn((1, 4)).to(device)
        
        with torch.no_grad():
          rec_img  = decoder(random_encoded_image)

        plt.imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')  
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)  
        if r == 0 and c == col//2:
         ax.set_title('Generated images')


    plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.7, 
                    top=0.9, 
                    wspace=0.3, 
                    hspace=0.3)
    
    plt.show()


gen_and_plot_random(encoder,decoder,device,row=3,col=3,noise_factor=0.3)


# clustering accuracy calculation

#1.obtain encoder outputs for all training sets
train_encoder_outputs = []

def obtain_encoder_outputs(encoder, device, train_data,noise_factor=0.3):
    encoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        for image, label in train_data:
            image = image.unsqueeze(0)
            image_noisy = add_noise(image, noise_factor)
            image_noisy = image_noisy.to(device)
            # Encode data
            encoded_data = encoder(image_noisy)
            train_encoder_outputs.append((encoded_data, label))

obtain_encoder_outputs(encoder=encoder, device=device, train_data=train_data, noise_factor=0.3)

encoded_images = [img.to("cpu").numpy() for (img, _) in train_encoder_outputs]
encoded_images = np.squeeze(np.array(encoded_images))
encoded_labels = np.array([label for (_, label) in train_encoder_outputs])

# k-means to get 10 clusters from 48000 training image dataset. k-means will give cluster integer[0-9] for each image.
k = 10
kmeans = KMeans(n_clusters=k)
kmeans.fit(encoded_images)
cluster_labels = kmeans.labels_

# match k-means labels to actual labels to get mapping

#From predicted to actual labeling
def get_mapping(actual_labels, cluster_labels):
  mapping = {}
  for i in range(k):
    index = np.where(cluster_labels == i,1,0)
    actual_labels_in_cluster = actual_labels[index==1]
    counts = np.bincount(actual_labels_in_cluster)
    num = counts.argmax()


    mapping[i] = num
  return mapping

mapping = get_mapping(encoded_labels, cluster_labels)
print(f"\nmapping: {mapping}")

for index, label in enumerate(cluster_labels):
    cluster_labels[index] = mapping[label]

from sklearn.metrics import accuracy_score

#accuracy
print("accuracy: ",end="")
print(accuracy_score(cluster_labels, encoded_labels))