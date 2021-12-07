open Regler.slx
set_param('Regler/Gain','Gain','Kr') 
modelworkspace = get_param('Regler','ModelWorkspace');
assignin(modelworkspace,'Kr',Simulink.Parameter(0.65));
set_param('Regler/Transfer Fcn','Numerator','N') 
model_workspace = get_param('Regler','ModelWorkspace');
assignin(model_workspace,'N',Simulink.Parameter([0.9 1]));
set_param('Regler/Transfer Fcn','Denominator','D') 
model__workspace = get_param('Regler','ModelWorkspace');
assignin(model__workspace,'D',Simulink.Parameter([0 0.1 1]));
set_param('Regler','ParameterArgumentNames','Kr,N,D');
