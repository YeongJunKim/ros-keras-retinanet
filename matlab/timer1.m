function timer1(obj, event)

    persistent firstrun;
    persistent step;
    persistent sub
    if(isempty(firstrun))
       firstrun = 1;
       
        step = 1;
        sub = rossubscriber('/detection/label',@Messagecallback);
    end
%     disp(step);
    step = step + 1;
end


function Messagecallback(src, msg)

    disp(msg);
%     disp(src)
end