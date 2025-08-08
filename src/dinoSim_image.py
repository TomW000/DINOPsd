from setup import dates, neurotransmitters, os, glob, np, plt, tqdm, patches 
from perso_utils import get_processed_image, get_prediction, get_bbox

def Images_DINOSim(threshold, n):

    for date in dates:
        date_name = os.path.basename(os.path.normpath(date))
        
        for neuro in neurotransmitters:
            print('Loading data')
            # Load files
            fnames = glob(os.path.join(date, neuro, '*.hdf*'))
            fnames.sort()

            # Skip if no data
            if len(fnames) == 0:
                print(f"Skipping {neuro} in {date_name} (no data)")
                continue
            
            image_list, cropped_image_list, coordinate_list, cropped_coordinate_list, crop_limmits_list = [], [], [], [], []
            for p in fnames:
                img_o, c1, c2, img_c, c1_c, c2_c, ax12, ax22 = get_processed_image(p)
                coordinate_list.append((0,c1,c2))
                cropped_coordinate_list.append((0,c1_c,c2_c))
                image_list.append(img_o)
                cropped_image_list.append(img_c)
                ax11, ax21 = c1_c+c1, c2_c+c2
                crop_limmits_list.append((ax11, ax12, ax21, ax22))
                
            dataset = np.stack(image_list)
            dataset_c = cropped_image_list #np.stack(cropped_image_list)
            coordinates = np.stack(coordinate_list)
            coordinates_c = np.stack(cropped_coordinate_list)

            predictions, bboxes_list, cropped_predictions, cropped_bboxes_list = [], [], [], []
            print('Done loading data')
            for k in tqdm(range(len(dataset[:10])), desc=f'Processing {neuro} from {date_name}'):
                pred = get_prediction(dataset[k], coordinates[k])
                bboxes_list.append(get_bbox(pred, threshold))
                predictions.append(pred)
                
                pred_c = get_prediction(dataset_c[k], coordinates_c[k])
                cropped_bboxes_list.append(get_bbox(pred_c, threshold))
                cropped_predictions.append(pred_c)
                
            for i in range(n):
                
                plt.figure(figsize=(15, 5), dpi=300)
                plt.suptitle(f"{neuro.capitalize()}-Sample {i+1}", fontsize=14)

                '''
                plt.subplot(231)
                plt.imshow(dataset[i][0,...], cmap='gray')
                plt.scatter(coordinates[i][1], coordinates[i][2], 
                            color='red', 
                            marker='x', 
                            s=100, 
                            label='Ground Truth')
                x1, x2, y1, y2 = bboxes_list[i]
                width, height = x2 - x1, y2 - y1        
                rect = patches.Rectangle(
                    (x1, y1), width, height, 
                    linewidth=2, edgecolor='blue', facecolor='none', 
                    label='Detection')
                plt.gca().add_patch(rect)
                plt.title("Original Image")
                plt.legend()


                plt.subplot(232)
                plt.imshow(1-predictions[i][0,...], cmap='magma')
                plt.colorbar(label='Normalized Distance - w/o cropping')
                plt.title("DINOv2 Distance Map")


                plt.subplot(233)
                plt.imshow(predictions[i][0,...] < threshold, cmap='gray')
                plt.title(f"Binary Mask (threshold={threshold}) - w/o cropping")


                plt.subplot(234)
                plt.imshow(dataset_c[i][0,...], cmap='gray')
                plt.scatter(coordinates_c[i][1], coordinates_c[i][2], 
                            color='red', 
                            marker='x', 
                            s=100, 
                            label='Ground Truth')


                plt.subplot(235)
                plt.imshow(1-cropped_predictions[i][0,...], cmap='magma')
                plt.colorbar(label='Normalized Distance - w/ cropping')
                plt.title("DINOv2 Distance Map")


                plt.subplot(236)
                X = np.zeros((130,130))
                ax11, ax12, ax21, ax22 = crop_limmits_list[i]
                #X[ax11:ax12, ax21:ax22] = cropped_predictions[i][0,...]
                plt.imshow(cropped_predictions[i][0,...] < threshold, cmap='gray')
                plt.title(f"Binary Mask (threshold={threshold}) - w/ cropping")
                '''

                plt.subplot(131)
                plt.imshow(dataset[i][0,...], cmap='gray')
                plt.scatter(coordinates[i][1], coordinates[i][2], 
                            color='red', 
                            marker='x', 
                            s=100, 
                            label='Prompt')
                x1, x2, y1, y2 = bboxes_list[i]
                width, height = x2 - x1, y2 - y1        
                rect = patches.Rectangle(
                    (x1, y1), width, height, 
                    linewidth=2, edgecolor='blue', facecolor='none', 
                    label='w/o')
                plt.gca().add_patch(rect)

                x1, x2, y1, y2 = cropped_bboxes_list[i]
                ax11, ax12, ax21, ax22 = crop_limmits_list[i]
                width, height = x2 - x1, y2 - y1        
                rect = patches.Rectangle(
                    (x1, y1), width, height, 
                    linewidth=2, edgecolor='green', facecolor='none', 
                    label='w/')
                plt.gca().add_patch(rect)
                plt.title("Image")
                plt.legend()


                plt.subplot(132)
                plt.imshow(predictions[i][0,...] < threshold, cmap='gray')
                plt.title(f"Mask (threshold={threshold})-w/o cropping")


                plt.subplot(133)
                X = np.zeros((130,130))
                #X[ax11:ax12, ax21:ax22] = cropped_predictions[i][0,...]
                plt.imshow(cropped_predictions[i][0,...] < threshold, cmap='gray')
                plt.title(f"Mask (threshold={threshold}) - w/ cropping")


                plt.tight_layout()
                plt.show()

if __name__=='__main__':
    threshold = input('Threshold: ')
    n = input('Number of examples displayed: ')
    Images_DINOSim(threshold=0.25, n=2)