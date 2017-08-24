package com.tmf.dml.facetracking;

import android.content.Context;
import android.content.Intent;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.os.AsyncTask;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.ProgressBar;
import android.widget.TextView;

import resources_manager.ImagesProvider;


public class MainActivity extends AppCompatActivity {
    private ImagesProvider imagesProvider;
    private ProgressBar progressBar;
    private TextView textView;


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

            imagesProvider = new ImagesProvider();
            imagesProvider.generateFiles(getResources(),
                    getDir("trainingDir", Context.MODE_PRIVATE));

            return true;
        }

        @Override
        protected void onPostExecute(Boolean result) {
            if (result) {
                textView.setText("Creation completed!");
                progressBar.setVisibility(View.INVISIBLE);
                Intent intent = new Intent(MainActivity.this, FdActivity.class);
                Bundle extrasFR = new Bundle();
                extrasFR.putSerializable("imgPro", imagesProvider);
                intent.putExtras(extrasFR);
                startActivity(intent);

                finish();
            }
        }
    }
}
