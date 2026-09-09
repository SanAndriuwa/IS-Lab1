function hsv_value = spalva_color(Im)
% Return the mean hue of the detected object.
% Im is an RGB image read with imread.
% The mask selects the object pixels used for the color feature.

BW = im2bw(rgb2gray(Im), 0.95);
BW = imfill(~BW, 'holes');
BW = imopen(BW, strel('disk', 12));

if ~any(BW(:))
    hsv_value = NaN;
    return;
end

hsvImage = rgb2hsv(Im);
hue = hsvImage(:, :, 1);
hsv_value = mean(hue(BW));
