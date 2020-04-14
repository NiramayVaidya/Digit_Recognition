import network2
import cv2
import numpy as np
import overfitting
import argparse
import gi
gi.require_version('Gtk', '3.0')
gi.require_version('Pango', '1.0')
from gi.repository import Gtk, Pango

def on_window_key_press_event(window, event):
    if event.keyval == 113: #ASCII for q
        Gtk.main_quit()

parser = argparse.ArgumentParser()
parser.add_argument('--file', '-f', type=str, required=True, \
        help='give relative path for file in place of FILE')
args = parser.parse_args()
filepath = args.file
filename = filepath.split('/')

#can also pass 0 for second argument
image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

'''
isn't working
https://github.com/skvark/opencv-python/issues/46
goto above link to see unresolved issue

cv2.namedWindow(filename[-1], cv2.WINDOW_NORMAL)
cv2.imshow(filename[-1], image)
key = cv2.waitKey(0) & 0xFF #passing zero makes it wait undefinitely for a key stroke
if key == 113: #ASCII decimal value for key 'q', on press, exit from image
    cv2.destroyAllWindows()
'''

#cv2.imwrite('../data/7-resized.png', image)
image = cv2.resize(image, (28, 28))
image = image.reshape(784, 1)
net_trained = network2.load('network2_params.json')
output = net_trained.feedforward(image)
output_digit = np.argmax(output)

'''
text = 'File name-\n' + filepath + '\n\nOutput layer activations-\n' + \
        str(output) + '\n\nRecognized digit-\n' + str(output_digit)
'''

filepath_new = filepath[:-4] + '_resized.jpg'

image = image.reshape(28, 28)
image = cv2.resize(image, (512, 512))
cv2.imwrite(filepath_new, image)

window = Gtk.Window(title=filename[-1])

img = Gtk.Image()
img.set_from_file(filepath_new)

text_buf = Gtk.TextBuffer()
text = '\nFile name-\n'
text_buf.set_text(text, length=len(text))
start_iter = text_buf.get_start_iter()
end_iter = text_buf.get_end_iter()
tag = text_buf.create_tag('format', foreground='red', weight=Pango.Weight.BOLD, \
        style=Pango.Style.ITALIC, underline=Pango.Underline.SINGLE, \
        size=2*10**4)
text_buf.apply_tag(tag, start_iter, end_iter)
text = filepath

'''
a mark needs to created because after calling text_buf.insert(), the iter
passed to it(in this case, the end_iter which represents the end of the previous
text added to the buffer and simultaneously the position in the buffer where
the next text is to be inserted) loses its reference position in the buffer
since content of the buffer changes
Since this start position of the newly inserted text in the buffer is required
for formatting, the mark preserves this position and then using this mark, an 
iter can be retrieved from the buffer at this position
'''

mark = text_buf.create_mark(None, end_iter, True)
text_buf.insert(end_iter, text, length=len(text))
start_iter = text_buf.get_iter_at_mark(mark)
end_iter = text_buf.get_end_iter()
tag = text_buf.create_tag('format_1', foreground='blue', size=1.5*10**4)
text_buf.apply_tag(tag, start_iter, end_iter)
text_buf.delete_mark(mark)
text = '\n\nOutput layer activations-\n'
mark = text_buf.create_mark(None, end_iter, True)
text_buf.insert(end_iter, text, length=len(text))
start_iter = text_buf.get_iter_at_mark(mark)
end_iter = text_buf.get_end_iter()
tag = text_buf.create_tag('format_2', foreground='red', weight=Pango.Weight.BOLD, \
        style=Pango.Style.ITALIC, underline=Pango.Underline.SINGLE, \
        size=2*10**4)
text_buf.apply_tag(tag, start_iter, end_iter)
text_buf.delete_mark(mark)
text = str(output)
mark = text_buf.create_mark(None, end_iter, True)
text_buf.insert(end_iter, text, length=len(text))
start_iter = text_buf.get_iter_at_mark(mark)
end_iter = text_buf.get_end_iter()
tag = text_buf.create_tag('format_3', foreground='blue', size=1.5*10**4)
text_buf.apply_tag(tag, start_iter, end_iter)
text_buf.delete_mark(mark)
text = '\n\nRecognized digit-\n'
mark = text_buf.create_mark(None, end_iter, True)
text_buf.insert(end_iter, text, length=len(text))
start_iter = text_buf.get_iter_at_mark(mark)
end_iter = text_buf.get_end_iter()
tag = text_buf.create_tag('format_4', foreground='red', weight=Pango.Weight.BOLD, \
        style=Pango.Style.ITALIC, underline=Pango.Underline.SINGLE, \
        size=2*10**4)
text_buf.apply_tag(tag, start_iter, end_iter)
text_buf.delete_mark(mark)
text = str(output_digit)
mark = text_buf.create_mark(None, end_iter, True)
text_buf.insert(end_iter, text, length=len(text))
start_iter = text_buf.get_iter_at_mark(mark)
end_iter = text_buf.get_end_iter()
tag = text_buf.create_tag('format_5', foreground='blue', size=1.5*10**4)
text_buf.apply_tag(tag, start_iter, end_iter)
text_buf.delete_mark(mark)

text_view = Gtk.TextView()
text_view.set_buffer(text_buf)
text_view.set_editable(False)
text_view.set_cursor_visible(False)

hbox = Gtk.HBox()
hbox.pack_start(img, True, True, 0)
hbox.pack_start(text_view, True, True, 0)

viewport = Gtk.Viewport()
viewport.add(hbox)
viewport.set_size_request(1400, 600)

vbox = Gtk.VBox()
vbox.pack_start(viewport, True, True, 0)

scroll_window = Gtk.ScrolledWindow()
scroll_window.add_with_viewport(vbox)

overfitting.make_plots('network2_overfitting.json', 30, vbox, \
        training_set_size=50000)

vbox.show_all()

window.connect('destroy', Gtk.main_quit)
window.connect('key-press-event', on_window_key_press_event)
window.add(scroll_window)
window.set_position(Gtk.WindowPosition.CENTER_ALWAYS)
window.resize(1400, 1400)
window.show_all()
Gtk.main()
