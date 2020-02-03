file =  readtable('train.csv');

%remove string coluns
cols_to_rm = [4 5 9 11 12];
file(:,cols_to_rm ) = [];
file

%covert table to matrix
data  = table2array(file);

%survive 
survive = data(data(:,2)==1,:);
not_survive = data(data(:,2)==0,:);



%survive(:,4) = survive(:,4).^2;
%not_survive(:,4) = not_survive(:,4).^2;


%x - survive
%y - not survive
x_1 = split(survive,1,-1,3,3);
x_2 = split(survive,1,-1,4,4);
x_3 = split(survive,1,-1,5,5);
x_sum = sum(survive(:,[3,4]),2);
x_sum(:,1) = x_sum(:,1).^3;

y_1 = split(not_survive,1,-1,3,3);
y_2 = split(not_survive,1,-1,4,4);
y_3 = split(not_survive,1,-1,5,5);
y_sum = sum(not_survive(:,[3,4]),2);
y_sum(:,1) = y_sum(:,1).^3;

%x_axis  = ones(342,1);
%x_axis2  = ones(549,1);



hold on
plot3(x_1,x_2,x_3,'*')
plot3(y_1,y_2,y_3,'+')
%scatter(x_survive,y_survive,'*')
%scatter(x_not_survive,y_not_survive,'.')
hold off
grid

function f = split(file,row_start,row_end,col_start,col_end);
    if row_end == -1
    f = file(row_start:end,col_start:col_end);
    else
    f = file(row_start:row_end,col_start:col_end);
    end
end