#include <thread>
#include <memory>
#include "yaml-cpp/yaml.h"

#include "AVGSlam.hpp"

#ifdef PANGOLIN_ACTIVE
    #include "gui/pangolinInterface.hpp"
    const bool PANGOLIN_INSTALLED = true;
#else
    const bool PANGOLIN_INSTALLED = false;
#endif


YAML::Node getDataConfig(YAML::Node sys_config)
{
    std::string dataset_name = sys_config["Dataset"].as<std::string>();
    return YAML::LoadFile("config/data_conf/" + dataset_name + ".yaml");
}

std::vector<int> getPlaylist(YAML::Node data_config)
{
    std::vector<int> playlist;
    const YAML::Node& all_seq_nr = data_config["Data.playlist"];

    for ( YAML::const_iterator it = all_seq_nr.begin(); it != all_seq_nr.end(); ++it )
    {
        const YAML::Node& seq_nr_node = *it;
        int seq_nr = seq_nr_node.as<int>();
        playlist.push_back(seq_nr);
    }

    return playlist;
}

int main()
{
    // Load config file
    YAML::Node sys_config = YAML::LoadFile("config/sys_conf/dev_config.yaml");
    YAML::Node data_config = getDataConfig(sys_config);
    std::vector<int> playlist = getPlaylist(data_config);

    for ( int seq_nr : playlist )
    {
        // Setting up output folder.
        std::string out_path = sys_config["Trck.out.path"].as<std::string>() + sys_config["Dataset"].as<std::string>() + "/" + std::to_string(seq_nr) + ".txt";
        if (!std::filesystem::is_directory(sys_config["Trck.out.path"].as<std::string>() + sys_config["Dataset"].as<std::string>()))
        {
            std::cout << "Creating output directory..." << std::endl;
            std::filesystem::create_directory(sys_config["Trck.out.path"].as<std::string>() + sys_config["Dataset"].as<std::string>());
        }

        std::shared_ptr<Sequencer3> seq = std::make_shared<Sequencer3>( sys_config, data_config, seq_nr, true );
        std::shared_ptr<AVGSlam> avg_slam = std::make_shared<AVGSlam>( sys_config, data_config, seq, out_path );

        bool GUI_show = sys_config["UI.GUI_show"].as<bool>();

        std::thread tracking_thread;
        if (PANGOLIN_INSTALLED && GUI_show)
        {
            #ifdef PANGOLIN_ACTIVE
            std::cout << "PANGOLING ACTIVE\n";
            tracking_thread = std::thread(&AVGSlam::run, avg_slam);

            std::shared_ptr<GUI> gui = std::make_shared<GUI>();
            gui->GUIConfigParser(sys_config);

            gui->run( avg_slam->getTracker(), seq );
            #endif
        }
        else
        {
            avg_slam->run();
        }



        if (PANGOLIN_INSTALLED && GUI_show)
        {
            #ifdef PANGOLIN_ACTIVE
            avg_slam->setShutdown(true);
            tracking_thread.join();
            #endif
        }

        std::cout << "######################################" << std::endl;
    }
}