clear all;
close all;


filename = 'test_rot_3d_plot_2.gif';

for i = 1:(video_frame_rate*indiv_audio_length)
    figure(i);
    view_az = -38 + i*(360/video_frame_rate*indiv_audio_length);
    set(gcf, 'WindowState', 'maximized');
    subplot(1,2,1);
    x = linspace(-2*pi,2*pi);
    y1 = sin(x);
    y2 = cos(x);
    plot(x,y1,x,y2)



    %frame = getframe(h); 

    %im = frame2im(frame);

    rp(1,:) = [3,3,2.5];
    rp(2,:) = [7,7,3];

    M = 2;
    h = subplot(1,2,2);
    plot3(rp(1,1),rp(1,2),rp(1,3),'x','MarkerSize',12);
    hold on;
    for mm = 2:M
        plot3(rp(mm,1),rp(mm,2),rp(mm,3),'x');
    end
    %plot3(sp_path(:,1),sp_path(:,2),sp_path(:,3),'r.');
    axis([0 10 0 10 0 5]);
    pbaspect([1,1,1]);
    h.View = [view_az,20];
    grid on;
    box on;
    axis square;
    hold off;

    %get(h)


    g = figure(i);

    frame = getframe(g);
    im = frame2im(frame); 
    [imind,cm] = rgb2ind(im,256);
    % Write to the GIF File 
    if i == 1
        imwrite(imind,cm,filename,'gif', 'Loopcount',inf); 
    else 
        imwrite(imind,cm,filename,'gif','DelayTime',0.06,'WriteMode','append'); 
    end 
end