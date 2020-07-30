rosshutdown;
pause(3);
rosinit("192.168.0.5");


tm = timer('BusyMode', 'drop', 'ExecutionMode', 'fixedRate', 'Period', 1, 'TimerFcn', {@timer1});
start(tm);
