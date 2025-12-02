#include "qCirno.h"

//Qt
#include <QMainWindow>
#include <QInputDialog>

//qCC_db
#include <ccPointCloud.h>
#include <ccPolyline.h>
#include <ccScalarField.h>

#include <ManualSegmentationTools.h>

#include <QProcess>
#include <QInputDialog>
#include <QFile>
#include <QDebug>
#include <QTextStream>

#include <QProgressDialog>
#include <QMessageBox>
#include <QApplication>
#include <QThread>

//pcl
// #include <eigen3/Eigen/src/Core/Matrix.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>




#include "vector"
#include "iostream"
#include "thread"


QuestionDialog::QuestionDialog(QWidget *parent)
    : QDialog(parent)
{
    setWindowTitle("Опрос");
    
    QGroupBox *groupBox = new QGroupBox("Пожалуйста, ответьте на вопросы:");
    
    // Создаем форму
    QFormLayout *formLayout = new QFormLayout();
    
    checkBox1 = new QCheckBox();
    checkBox2 = new QCheckBox();
    checkBox3 = new QCheckBox();
    
    formLayout->addRow("CarCloud", checkBox1);
    formLayout->addRow("RoadCloud", checkBox2);
    formLayout->addRow("StaticCloud", checkBox3);
    
    groupBox->setLayout(formLayout);
    
    // Кнопки
    QDialogButtonBox *buttonBox = new QDialogButtonBox(
        QDialogButtonBox::Ok | QDialogButtonBox::Cancel);
    
    connect(buttonBox, &QDialogButtonBox::accepted, this, &QDialog::accept);
    connect(buttonBox, &QDialogButtonBox::rejected, this, &QDialog::reject);
    
    QVBoxLayout *mainLayout = new QVBoxLayout(this);
    mainLayout->addWidget(groupBox);
    mainLayout->addWidget(buttonBox);
}


bool QuestionDialog::isFirstChecked() const
{
    return checkBox1->isChecked();
}

bool QuestionDialog::isSecondChecked() const
{
    return checkBox2->isChecked();
}

bool QuestionDialog::isThirdChecked() const
{
    return checkBox3->isChecked();
}



qCirnoPlugin::qCirnoPlugin(QObject* parent)
	: QObject(parent)
	, ccStdPluginInterface( ":/CC/plugin/qCirnoPlugin/info.json" )
	, m_action(nullptr)
{

}

bool hasClouds(const ccHObject::Container& selectedEntities)
{
	// проверяем что выбранно облако 
	// проверка с ошибками в бущем исправить
	for(int i = 0; i < selectedEntities.size(); i++)
	{
		if(selectedEntities[i]->isA(CC_TYPES::POINT_CLOUD))
		{
			return true;
		}
	}
	return false;
}


void qCirnoPlugin::onNewSelection(const ccHObject::Container& selectedEntities)
{
	if (m_action)
	{
		m_action->setEnabled(hasClouds(selectedEntities) );
	}

	m_selectedEntities = selectedEntities;
}


QList<QAction *> qCirnoPlugin::getActions()
{
	if (!m_action)
	{
		m_action = new QAction(getName(),this);
		m_action->setToolTip(getDescription());
		m_action->setIcon(getIcon());
		connect(m_action, &QAction::triggered, this, &qCirnoPlugin::doAction);
	}

	return QList<QAction *>{ m_action };
}


bool qCirnoPlugin::prepareData()
{
	if(m_selectedEntities.size() == 0)
	{
		return false;
	}

	m_clouds.clear();

	for(int i = 0; i < m_selectedEntities.size(); i++)
	{
		if(m_selectedEntities[i]->isA(CC_TYPES::POINT_CLOUD))
		{
			m_clouds.push_back(ccHObjectCaster::ToPointCloud(m_selectedEntities[i]));
		}
	}

	return true;
}


void qCirnoPlugin::doAction()
{

	QWidget* mainWindow = m_app ? m_app->getMainWindow() : nullptr;

	QuestionDialog dialog(mainWindow);

	bool Car;
	bool Road;
    bool Static;

	 if (dialog.exec() == QDialog::Accepted) {
        Car    = dialog.isFirstChecked();
        Road   = dialog.isSecondChecked();
        Static = dialog.isThirdChecked();
	 }

	if(!prepareData())
	{
		return;
	}

	bool ok = true;

	QProgressDialog progress("Выполнение операции", "Отмена", 0, 100);
    progress.setWindowTitle("Ожидание");
    progress.setWindowModality(Qt::WindowModal);
    progress.show();

    QApplication::processEvents();

	
  	std::cerr << "BAKA Start" << std::endl; 

	pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

	int sfIndex = m_clouds[0]->getScalarFieldIndexByName("intensity");

	for(int i = 0; i < m_clouds[0]->size(); ++i)
		{
			pcl::PointXYZI point;
			point.x = m_clouds[0]->getPoint(i)->x;
			point.y = m_clouds[0]->getPoint(i)->y;
			point.z = m_clouds[0]->getPoint(i)->z;
			point.intensity = m_clouds[0]->getPointScalarValue(i);

			cloud->push_back(point);
		}

	pcl::io::savePCDFileBinary ("/app/Share/myKPConv/data/cloud/cloud.pcd", *cloud);


	QString program = "python3";

	// Запустим в работу Нейронку
	QString nn_alg =  "/app/Share/myKPConv/model.py";
	QStringList nn_arguments;
	nn_arguments << nn_alg;

	QProcess nn_process;
	nn_process.setWorkingDirectory("/app/Share/myKPConv");


	nn_process.setProcessChannelMode(QProcess::MergedChannels);
	int num_progress;

	QObject::connect(&nn_process, &QProcess::readyReadStandardOutput, [&]() {

		if (progress.wasCanceled()) {
            return;
        }

		QByteArray output = nn_process.readAllStandardOutput();

		std::string text = output.toStdString();
		std::string progress_count (text.begin(), text.begin() + 4);
		std::stringstream ss(progress_count);

		if (!progress_count.empty())
				ss >> num_progress;

		progress.setValue(num_progress);
        progress.setLabelText(QString("Выполнение... %1%").arg(num_progress));
		QApplication::processEvents();
		
		std::cout << output.toStdString() << std::flush;
	});

	nn_process.start(program, nn_arguments);

	if (nn_process.waitForStarted()) {
		std::cerr << "Process STARTED successfully" << std::endl;
		
		// Ждем завершения
		if (nn_process.waitForFinished(-1)) {  //600000
			std::cerr << "Process FINISHED with exit code: " << nn_process.exitCode() << std::endl;
			progress.close();
			
			// Прочитаем остатки вывода
			QByteArray finalOutput = nn_process.readAll();
			if (!finalOutput.isEmpty()) {
				std::cout << "FINAL OUTPUT: " << finalOutput.toStdString() << std::endl;
			}
		} else {
			std::cerr << "Process failed to finish properly" << std::endl;
		}
	} else {
		std::cerr << "Failed to START process" << std::endl;
	}

	if (Car)
	{
		pcl::PointCloud<pcl::PointXYZI>::Ptr CarCloud(new pcl::PointCloud<pcl::PointXYZI>);
		pcl::io::loadPCDFile<pcl::PointXYZI> ("/app/Share/myKPConv/CAR_cloud.pcd", *CarCloud); //* load the file 

		ccPointCloud* CarCCloud = createFromPCL(CarCloud, "CarCloud");
		m_app->addToDB(CarCCloud);
	}

	if (Road)
	{
	pcl::PointCloud<pcl::PointXYZI>::Ptr RoadCloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::io::loadPCDFile<pcl::PointXYZI> ("/app/Share/myKPConv/Road_cloud.pcd", *RoadCloud); //* load the file 

	ccPointCloud* RoadCCloud = createFromPCL(RoadCloud, "RoadCloud");
	m_app->addToDB(RoadCCloud);
	}


	if (Static)
	{
	pcl::PointCloud<pcl::PointXYZI>::Ptr StaticCloud(new pcl::PointCloud<pcl::PointXYZI>);
	pcl::io::loadPCDFile<pcl::PointXYZI> ("/app/Share/myKPConv/Static_obj_cloud.pcd", *StaticCloud); //* load the file 

	ccPointCloud* StaticCCloud = createFromPCL(StaticCloud, "StaticCloud");
	m_app->addToDB(StaticCCloud);
	}

	std::cerr << "BAKA End" << std::endl; 


	if(!ok)
	{
		return;
	}
}


ccPointCloud* qCirnoPlugin::createFromPCL(const pcl::PointCloud<pcl::PointXYZI>::Ptr& pclCloud, QString name)
{
    ccPointCloud* cloud = new ccPointCloud(name);
	std::cerr << "pclCloud->size()" << pclCloud->size() << std::endl; 

    // Добавляем точки и интенсивность
    for (size_t i = 0; i < pclCloud->size(); ++i)
    {
        const auto& pclPoint = pclCloud->at(i);
        CCVector3 point(pclPoint.x, pclPoint.y, pclPoint.z);

		cloud->addPoint(point);
    }


	    // Создаем скалярное поле для интенсивности
    int intensityIndex = cloud->addScalarField("Intensity");
    CCCoreLib::ScalarField* intensitySF = cloud->getScalarField(intensityIndex);

	    // Заполняем скалярное поле
    for (size_t i = 0; i < pclCloud->size(); ++i)
    {
        const auto& pclPoint = pclCloud->at(i);
        intensitySF->setValue(static_cast<int>(i), static_cast<ScalarType>(pclPoint.intensity));
    }


	std::cerr << "size cloud " << cloud->size() << std::endl; 
	std::cerr << "Scalar Field " << intensitySF->size() << std::endl; 
    

    intensitySF->computeMinAndMax();

    cloud->setCurrentDisplayedScalarField(intensityIndex);
    cloud->showSF(true);

        // 6. Обновляем отображение
        cloud->colorsHaveChanged();
        cloud->prepareDisplayForRefresh();

    
    cloud->invalidateBoundingBox();
    return cloud;
}


void qCirnoPlugin::registerCommands(ccCommandLineInterface* cmd)
{
}
