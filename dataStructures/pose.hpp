#ifndef pose_h
#define pose_h

#include "frameData.hpp"
#include "parametrization.hpp"

// Forward declaration for circular dependence
class FrameData;

class Pose
{
    private:
        int pose_nr;
        cv::Mat T_matrix;
        cv::Mat R_matrix;
        cv::Mat t_vector;
        cv::Mat E_matrix;
        std::weak_ptr<FrameData> frame1;
        std::weak_ptr<FrameData> frame2;
        std::map<ParamID, std::shared_ptr<Parametrization>> params;

        // Mutexes
        mutable std::shared_mutex mutex_pose;
        mutable std::shared_mutex mutex_params_map;

        // Write functions
        void setTMatrix(cv::Mat T_matrix);
        void setRMatrix(cv::Mat R_matrix);
        void settvector(cv::Mat t_vector);
        void setEmatrix(cv::Mat E_matrix);
        void setParametrization( std::vector<double> params, ParamID parametrization_id=ParamID::STDPARAM );
        void createParametrization( std::vector<double> params, ParamID parametrization_id=ParamID::STDPARAM );
        void createParametrization( cv::Mat R, cv::Mat t, ParamID parametrization_id=ParamID::STDPARAM );
    
    public:
        Pose( std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, int pose_nr=-1 );
        Pose( cv::Mat E_matrix, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, int pose_nr=-1 );
        //Pose( cv::Mat T_matrix, std::shared_ptr<FrameData> frame1, std::shared_ptr<FrameData> frame2, int pose_nr=-1 );
        ~Pose();

        // Write functions
        void setPose( cv::Mat E_matrix );
        void setPose( std::vector<double> params, ParamID parametrization_id=ParamID::STDPARAM );
        void updateParametrization( ParamID parametrization_id=ParamID::STDPARAM );
        void updatePoseVariables(cv::Mat T_matrix);
        void updatePoseVariables(cv::Mat R_matrix, cv::Mat t_vector);
        void updatePoseVariables(cv::Mat E_matrix, cv::Mat R_matrix, cv::Mat t_vector);
        void updatePoseVariables(ParamID parametrization_id);
        void riseParametrizationInvalidFlags(ParamID exeption=ParamID::NONE);


        // Read functions
        int getPoseNr();
        std::shared_ptr<FrameData> getFrame1();
        std::shared_ptr<FrameData> getFrame2();
        cv::Mat getTMatrix();
        cv::Mat getRMatrix();
        cv::Mat gettvector();
        cv::Mat getEMatrix();
        cv::Mat calculateFMatrix(cv::Mat K_inv);
        std::shared_ptr<FrameData> getConnectingFrame( int connecting_frame_nr );
        std::shared_ptr<Parametrization> getParametrization( ParamID parametrization_id=ParamID::STDPARAM );
        void decomposeEMatrixSlow( cv::Mat &E_matrix, cv::Mat &R, cv::Mat &t);
        void write2File( std::string file_name, ParamID parametrization_id=ParamID::STDPARAM );

        // Static functions
        static cv::Mat composeTMatrix( cv::Mat R_matrix, cv::Mat t_vector );    //TODO: Change these to input reference
        static cv::Mat composeEMatrix( cv::Mat R, cv::Mat t );

        // Operator overload
        friend std::ostream& operator<<(std::ostream& os, const Pose& obj);
};

#endif