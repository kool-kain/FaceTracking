package com.tmf.dml.facetracking;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.os.AsyncTask;
import android.os.Bundle;
import android.os.Handler;
import android.support.v7.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

import data.FaceRecognizerElements;
import resources_manager.ImagesProvider;

import static org.bytedeco.javacpp.opencv_face.FaceRecognizer;
import static org.bytedeco.javacpp.opencv_face.createFisherFaceRecognizer;

public class MainActivity extends AppCompatActivity {
    private static final String TAG = "MainActivity::Activity";
    private static final double THERESOLD = 90;

    private FaceRecognizer faceRecognizerFace;
    private FaceRecognizer faceRecognizerEmotion;
    private ImagesProvider imagesProvider;
    private ProgressBar progressBar;
    private TextView textView;
    private Bundle extrasFR;
    private Handler handler = new Handler();


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        textView = (TextView) findViewById(R.id.textView);
        textView.setText("Creating database... Please wait");
        progressBar = (ProgressBar) findViewById(R.id.progressBar);
        progressBar.getIndeterminateDrawable()
                .setColorFilter(Color.BLUE, PorterDuff.Mode.SRC_IN);
        progressBar.setVisibility(View.VISIBLE);
        InitTask loadRecognizers = new InitTask();
        loadRecognizers.execute();

    }

    private class InitTask extends AsyncTask<Void, Void, Boolean> {

        @Override
        protected Boolean doInBackground(Void... voids) {

            imagesProvider = new ImagesProvider(getResources(),
                    getDir("trainingDir", Context.MODE_PRIVATE));

            faceRecognizerFace = createFisherFaceRecognizer();
            faceRecognizerFace.setThreshold(THERESOLD);
            faceRecognizerEmotion = createFisherFaceRecognizer();
            faceRecognizerEmotion.setThreshold(THERESOLD + 10);
            initTrainers();

            return true;
        }

        @Override
        protected void onPostExecute(Boolean result) {
            if (result) {
                textView.setText("Creation completed!");
                progressBar.setVisibility(View.INVISIBLE);
                Intent intent = new Intent(MainActivity.this, FdActivity.class);
                extrasFR = new Bundle();
                extrasFR.putLong("faceRFaces", faceRecognizerFace.address());
                extrasFR.putLong("faceREmotions", faceRecognizerEmotion.address());
                Log.d(TAG, "Fa - " + faceRecognizerFace.address() + " || Fe - " + faceRecognizerEmotion.address());
                intent.putExtras(extrasFR);
                startActivity(intent);
                faceRecognizerFace.clear();
                faceRecognizerFace.close();
                faceRecognizerFace.deallocate();
                faceRecognizerEmotion.clear();
                faceRecognizerEmotion.close();
                faceRecognizerEmotion.deallocate();
                finish();
            }

        }
    }

    private void initTrainers() {
        //First we recognized gender by training with a vector of labelled faces
        FaceRecognizerElements faceRecognizerElements = imagesProvider.getAllImagesGenderLabelled();
        if (faceRecognizerElements != null && faceRecognizerElements.getMatVector() != null
                && faceRecognizerElements.getLabels() != null) {
            faceRecognizerFace.train(faceRecognizerElements.getMatVector(),
                    faceRecognizerElements.getLabels());
        } else {
            Log.d(TAG, "Couldn't training faces model");
        }

        //And we recognized emotions by training with a vector of labelled emotion
        FaceRecognizerElements faceRecognizerElementsEmotions = imagesProvider.getAllImagesEmotionsLabelled();
        if (faceRecognizerElementsEmotions != null && faceRecognizerElementsEmotions.getMatVector() != null
                && faceRecognizerElementsEmotions.getLabels() != null) {
            faceRecognizerEmotion.train(faceRecognizerElementsEmotions.getMatVector(),
                    faceRecognizerElementsEmotions.getLabels());
        } else {
            Log.d(TAG, "Couldn't training emotions model");
        }
    }
}
