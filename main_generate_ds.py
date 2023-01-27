from PIL import Image

if __name__ == '__main__':
    img = Image.open(r"./input_dataset/test.jpg")
    for i in range(40):
        img.save(f'./input_dataset/test{i}.jpg', 'jpeg')
