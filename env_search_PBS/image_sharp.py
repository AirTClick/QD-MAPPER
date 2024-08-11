from PIL import Image, ImageFilter

# Load an image from file
image = Image.open('unrepaired.png')

# Apply sharpen filter
sharpened_image = image.filter(ImageFilter.SHARPEN)

# Save the sharpened image
sharpened_image.save('unrepaired_.png')