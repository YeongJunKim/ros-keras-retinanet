function timer1(obj, event)

    persistent firstrun;
    persistent step;
    persistent sub
    if(isempty(firstrun))
       firstrun = 1;
       
        step = 1;
        sub = rossubscriber('/detection/label',@Messagecallback);
    end
%      disp(step);
    step = step + 1;
end


function Messagecallback(src, msg)

%     disp(msg);
    detectiondata = zeros(4, msg.LabelNum);
    
    for i = 1:msg.LabelNum
       detectiondata(:,i) = msg.Data(4*(1-i)+1:4*(1-i)+4);
    end
    fprintf("[seq %d] X axis size : %d, Y axis size :%d",msg.Seq, msg.XAxis, msg.YAxis);
    detectiondata
    
%     disp(src)
end