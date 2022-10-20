#ifndef parametrization_h
#define parametrization_h

#include <shared_mutex>

enum class ParamID {NONE, STDPARAM, LIEPARAM};

class Parametrization
{
    private:
        bool valid;
        ParamID parametrization_id;

        // Mutexes
        mutable std::shared_mutex mutex_valid_flag;
    
    protected:
        mutable std::shared_mutex mutex_parametrization;

    public:
        Parametrization(ParamID parametrization_id, bool valid);
        ~Parametrization();

        // Write functions
        void setValidFlag(bool value);
        virtual void setParams(std::vector<double> params)=0;
        virtual void setParams( cv::Mat R, cv::Mat t )=0;
        virtual void decomposeRMatrix( cv::Mat &R )=0;
        virtual void decomposeTVector( cv::Mat t )=0;

        // Read functions
        bool isValid();
        virtual std::vector<double> getRotParams()=0;
        virtual std::vector<double> getTransParams()=0;
        virtual std::vector<double> getParamVector()=0;
        virtual cv::Mat composeRMatrix()=0;
        virtual cv::Mat composeTransVec()=0;
        virtual std::ostream& print(std::ostream& out)=0;


        // Static functions
        static bool isRotationMatrix(cv::Mat &R);
        virtual void composeRMatrixAndTParam(std::vector<double>& param, cv::Mat& R, cv::Mat& t)=0;

        // Operator overload
        friend std::ostream& operator<<(std::ostream &out, Parametrization &obj);
};


class StdParam : public Parametrization
{
    private:
        double rx, ry, rz, tx, ty, tz;
    
    public:
        StdParam();
        StdParam(std::vector<double> params, bool valid=true);
        StdParam(double rx, double ry, double rz, double tx, double ty, double tz, bool valid=true);
        StdParam(cv::Mat R, cv::Mat t, bool valid=true);
        ~StdParam();

        // Write functions
        void setParams(std::vector<double> params)override;
        void setParams( cv::Mat R, cv::Mat t )override;
        void decomposeRMatrix( cv::Mat &R )override;
        void decomposeTVector( cv::Mat t )override;
        
        // Read functinos
        std::vector<double> getRotParams()override;
        std::vector<double> getTransParams()override;
        std::vector<double> getParamVector()override;
        cv::Mat composeRMatrix()override;
        cv::Mat composeTransVec()override;
        std::ostream& print(std::ostream& out)override;

        // Static functions
        void composeRMatrixAndTParam(std::vector<double>& param, cv::Mat& R, cv::Mat& t)override;
};

class LieParam : public Parametrization
{
    private:
        double w1, w2, w3, tx, ty, tz;
    
    public:
        LieParam();
        LieParam(std::vector<double> params, bool valid=true);
        LieParam(double rx, double ry, double rz, double tx, double ty, double tz, bool valid=true);
        LieParam(cv::Mat R, cv::Mat t, bool valid=true);
        ~LieParam();

        // Write functions
        void setParams(std::vector<double> params)override;
        void setParams( cv::Mat R, cv::Mat t )override;
        void decomposeRMatrix( cv::Mat &R )override;
        void decomposeTVector( cv::Mat t )override;
        
        // Read functinos
        std::vector<double> getRotParams()override;
        std::vector<double> getTransParams()override;
        std::vector<double> getParamVector()override;
        cv::Mat composeRMatrix()override;
        cv::Mat composeTransVec()override;
        std::ostream& print(std::ostream& out)override;

        // Static functions
        void composeRMatrixAndTParam(std::vector<double>& param, cv::Mat& R, cv::Mat& t)override;
};

//TODO: Give warning if user tries to use non-valid parameters.
#endif
