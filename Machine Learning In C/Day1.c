#include <stdio.h>
float train[][2]={
    {0.0,1.0},
    {1.0,3.0},
    {2.0,5.0},
    {3.0,7.0},
    {4.0,9.0},
    {5.0,11.0}
};
int main(){
    int n=6; // 6 features
    int espochs=100000;
    // predict y=wx+b , a=? b=?
    float w=0.0;
    float b=0.0;
    float learning_rate=0.001;
    for (int espoch=0;espoch<espochs;espoch++){
        float total_loss=0.0;
        for (int i=0;i<n;i++){
            float x=train[i][0];
            float y=train[i][1];
            float y_predict = x*w+b;
            // error=y_predict - y
            float error = y_predict-y;
            // loss = |error| ,error*erorr
            float loss = (error)*(error);
            total_loss+=loss;
            // goal : minimize loss , loss =( ax+b-y)^2,
            float grad_w=2*x*(error);
            float grad_b=2*error;
            w= w-learning_rate*grad_w;
            b= b-learning_rate*grad_b;
        }
    if (espoch%100==0){
        printf("Espoch : %d || Loss = %f || w = %f || b = %f \n",espoch,total_loss,w,b);  
    }
}
     float test_x=7.0;
    printf("Mo hinh la : y= %f*x+ %f , y(7) := %f",w,b,test_x*w+b);
}