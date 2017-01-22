// code based on https://github.com/BVLC/caffe/wiki/Simple-Example:-Sin-Layer

#include <vector>

#include "caffe/caffe.hpp"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
//#include "caffe/layers/argmax_layer.hpp"
#include "sin_layer.hpp"
#include "caffe/test/test_gradient_check_util.hpp"
using namespace caffe;
using namespace std;

typedef double Dtype;

int main(int argc, char** argv) {

    Caffe::set_mode(Caffe::CPU);
    Caffe::set_random_seed(1701);

    // bottom/top blobs
    Blob<Dtype>* blob_top = new Blob<Dtype>();
    Blob<Dtype>* blob_bottom = new Blob<Dtype>(10, 20, 1, 1);

    // blob vector
    vector<Blob<Dtype>*> blob_bottom_vec;
    vector<Blob<Dtype>*> blob_top_vec;
    blob_bottom_vec.push_back(blob_bottom);
    blob_top_vec.push_back(blob_top);

    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(blob_bottom);

    ///////////////////////////////////////////
    // TestForward
    LayerParameter layer_param;
    SinLayer<Dtype> layer(layer_param);
    layer.SetUp(blob_bottom_vec, blob_top_vec);
    layer.Forward(blob_bottom_vec, blob_top_vec);

    // Now, check values
    const Dtype* bottom_data = blob_bottom->cpu_data();
    const Dtype* top_data = blob_top->cpu_data();
    //const Dtype min_precision = 1e-5;
    cout << "\ttop_data\texpected_value"<<endl;
    for (int i = 0; i < blob_bottom->count(); ++i)
    {
        Dtype expected_value = sin(bottom_data[i]);
        //Dtype precision = std::max(Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
        //EXPECT_NEAR(expected_value, top_data[i], precision);
        cout <<"#"<< i<<"\t"<< top_data[i] <<"\t" <<expected_value << endl;
    }

    cout << blob_top->count() << " samples using "<< layer.type()<< " layer"<< endl;


#ifdef DO_IT_LATER
    ///////////////////////////////////////////
    // TestBackward
    LayerParameter layer_param1;
    SinLayer<Dtype> layer1(layer_param1);
    GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
    checker.CheckGradientEltwise(&layer1, blob_bottom_vec, blob_top_vec);
#endif


    return 0;
}


