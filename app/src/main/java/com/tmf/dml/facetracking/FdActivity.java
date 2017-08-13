package com.tmf.dml.facetracking;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class FdActivity extends Activity implements CvCameraViewListener2 {

    private static final String TAG = "OCVSample::Activity";
    private static final Scalar FACE_RECT_COLOR = new Scalar(0, 255, 0);
    public static final int JAVA_DETECTOR = 0;
    public static final int NATIVE_DETECTOR = 1;
    public static final float EYE_SX = 0.16f;
    public static final float EYE_SY = 0.26f;
    public static final float EYE_SW = 0.30f;
    public static final float EYE_SH = 0.38f;

    private MenuItem mItemFace50;
    private MenuItem mItemFace40;
    private MenuItem mItemFace30;
    private MenuItem mItemFace20;
    private MenuItem mItemType;
    private Mat matRgba;
    private Mat matGray;
    private Mat matDest;
    private File fileCascadeFile;
    private CascadeClassifier cascadeClassifierFace, cascadeClassifierEye;
    private int detectorType = JAVA_DETECTOR;
    private String[] detectorName;
    private float relativeFaceSize = 0.2f;
    private int absoluteFaceSize = 0;

    private CameraBridgeViewBase camOpenCvCameraView;

    private BaseLoaderCallback baseLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    Log.i(TAG, "OpenCV loaded successfully");

                    try {
                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
                        fileCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
                        FileOutputStream os = new FileOutputStream(fileCascadeFile);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();

                        cascadeClassifierFace = new CascadeClassifier(fileCascadeFile.getAbsolutePath());
                        if (!cascadeClassifierFace.load(fileCascadeFile.getAbsolutePath())) {
                            Log.d(TAG, "Failed to load cascade classifier");
                            cascadeClassifierFace = null;
                        } else {
                            Log.d(TAG, "Loaded cascade classifier from " + fileCascadeFile.getAbsolutePath());
                        }

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Unable to loading classifier. Exception thrown: " + e);
                    }

                    camOpenCvCameraView.enableView();
                }
                break;
                default: {
                    super.onManagerConnected(status);
                }
                break;
            }
        }
    };

    public FdActivity() {
        detectorName = new String[2];
        detectorName[JAVA_DETECTOR] = "Java";
        detectorName[NATIVE_DETECTOR] = "Native (tracking)";

        Log.i(TAG, "Instantiated new " + this.getClass());
    }

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setContentView(R.layout.face_detect_surface_view);

        camOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.fd_activity_surface_view);
        camOpenCvCameraView.setVisibility(CameraBridgeViewBase.VISIBLE);
        camOpenCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onPause() {
        super.onPause();
        if (camOpenCvCameraView != null)
            camOpenCvCameraView.disableView();
    }

    @Override
    public void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, baseLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            baseLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        camOpenCvCameraView.disableView();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        matGray = new Mat();
        matDest = new Mat();
        matRgba = new Mat();
    }

    @Override
    public void onCameraViewStopped() {
        matGray.release();
        matDest.release();
        matRgba.release();
    }

    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {

        matRgba = inputFrame.rgba();
        matGray = inputFrame.gray();
        matDest = matRgba.clone();

        Imgproc.equalizeHist(matGray, matGray);

        MatOfRect faces = new MatOfRect();

        if (absoluteFaceSize == 0) {
            int height = matGray.rows();
            if (Math.round(height * relativeFaceSize) > 0) {
                absoluteFaceSize = Math.round(height * relativeFaceSize);
            }
        }

        if (cascadeClassifierFace != null) {
            cascadeClassifierFace.detectMultiScale(matGray, faces, 1.1, 4, 0,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        } else {
            Log.d(TAG, "Missed classifier, therefore no face detection allowed");
        }

        Log.d(TAG, "Faces length: " + faces.toArray().length + ". AbsFaceSize: " + absoluteFaceSize);

        for (Rect rect : faces.toArray()) {
            Imgproc.rectangle(matDest, rect.tl(), rect.br(), FACE_RECT_COLOR, 2);
        }

        return matDest;
    }


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        mItemFace50 = menu.add("Face size 50%");
        mItemFace40 = menu.add("Face size 40%");
        mItemFace30 = menu.add("Face size 30%");
        mItemFace20 = menu.add("Face size 20%");
        mItemType = menu.add(detectorName[detectorType]);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.d(TAG, "called onOptionsItemSelected; selected item: " + item);
        if (item == mItemFace50)
            setMinFaceSize(0.5f);
        else if (item == mItemFace40)
            setMinFaceSize(0.4f);
        else if (item == mItemFace30)
            setMinFaceSize(0.3f);
        else if (item == mItemFace20)
            setMinFaceSize(0.2f);

        return true;
    }

    private void setMinFaceSize(float faceSize) {
        relativeFaceSize = faceSize;
        absoluteFaceSize = 0;
    }
}
