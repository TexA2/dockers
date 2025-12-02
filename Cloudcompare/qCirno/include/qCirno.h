#ifndef Q_CIRNO_PLUGIN_HEADER
#define Q_CIRNO_PLUGIN_HEADER

//qCC
#include "ccStdPluginInterface.h"

//qCC_db
#include <ccHObject.h>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <thread>
#include <mutex>
#include <atomic>


#include <QDialog>
#include <QCheckBox>
#include <QLabel>
#include <QDialogButtonBox>
#include <QVBoxLayout>
#include <QGroupBox>
#include <QFormLayout>


using namespace std::chrono_literals; // Для использования литералов времени


class qCirnoPlugin : public QObject , public ccStdPluginInterface
{
    Q_OBJECT
    Q_INTERFACES( ccPluginInterface ccStdPluginInterface )

    Q_PLUGIN_METADATA( IID "cccorp.cloudcompare.plugin.qCirno" FILE "../info.json" )

public:

	//! Default constructor
	qCirnoPlugin(QObject* parent = nullptr);

	virtual ~qCirnoPlugin() = default;

	//inherited from ccStdPluginInterface
	virtual void onNewSelection(const ccHObject::Container& selectedEntities) override;
	virtual QList<QAction *> getActions() override;
	virtual void registerCommands(ccCommandLineInterface* cmd) override;
	ccPointCloud* createFromPCL(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pclCloud, QString name = "PCL Cloud");

private:

	void doAction();
	bool prepareData();

	//! Default action
	QAction* m_action;

	//! Currently selected entities
	ccHObject::Container m_selectedEntities;

	std::vector<ccPointCloud*> m_clouds;

	std::mutex mtx; 				  // Мьютекс для синхронизации доступа к общим данным
    std::atomic<int> clusterID  {1};  // Атомарная переменная для ID кластера
};



class QuestionDialog : public QDialog
{
    Q_OBJECT

public:
    explicit QuestionDialog(QWidget *parent = nullptr);
    
    // Методы для получения состояния чекбоксов
    bool isFirstChecked() const;
    bool isSecondChecked() const;
    bool isThirdChecked() const;
    
private:
    QCheckBox *checkBox1;
    QCheckBox *checkBox2;
    QCheckBox *checkBox3;
};



#endif
