// This file can be replaced during build by using the `fileReplacements` array.
// `ng build --prod` replaces `environment.ts` with `environment.prod.ts`.
// The list of file replacements can be found in `angular.json`.
// firebase functions:config:set anthropic.apikey="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"
// firebase functions:config:set openai.apikey="sk-None-HHxqTNa7lG5ANTLSdpmzT3BlbkFJHd0PY5GKdiOHFAE0dEi7"
// firebase functions:config:set pinecone.apikey="82d78500-b9e4-4574-b1e4-8fbc3764e698"

export const environment = {
  production: false,
  firebase: {
    apiKey: "AIzaSyAmR16UZg8KAzj7q0mroPR2lDm0sTlwuo4",
    authDomain: "alfred-d34e3.firebaseapp.com",
    projectId: "alfred-d34e3",
    storageBucket: "alfred-d34e3.appspot.com",
    messagingSenderId: "958181082962",
    appId: "1:958181082962:web:135c6fd7ade512410b9c8b",
    measurementId: "G-X5M31WBVGL"
  }
};

/*
 * For easier debugging in development mode, you can import the following file
 * to ignore zone related error stack frames such as `zone.run`, `zoneDelegate.invokeTask`.
 *
 * This import should be commented out in production mode because it will have a negative impact
 * on performance if an error is thrown.
 */
// import 'zone.js/dist/zone-error';  // Included with Angular CLI.
