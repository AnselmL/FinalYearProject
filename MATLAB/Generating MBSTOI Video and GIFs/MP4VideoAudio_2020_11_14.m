%[~,videoObj] = gif2avi('MBSTOI_static_changing_angle.gif','.mp4');

clear all;

videoFReader = vision.VideoFileReader('MBSTOI_static_changing_angle.mp4');
videoFWriter = vision.VideoFileWriter('myFile.mp4', 'FileFormat', 'MPEG4',...
    'FrameRate',1);


for i=1
  videoFrame = videoFReader();
  videoFWriter(videoFrame);
end

release(videoFReader);
release(videoFWriter);

