rosshutdown;
% delete_timer();
pause(3);
rosinit("192.168.0.3");
pause(1);
tm = timer('BusyMode', 'drop', 'ExecutionMode', 'fixedRate', 'Period', 1, 'TimerFcn', {@timer1});
start(tm);
