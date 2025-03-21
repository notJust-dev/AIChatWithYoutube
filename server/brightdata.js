const brightDataTriggerUrl = 'https://api.brightdata.com/datasets/v3/trigger';
const webhookUrl = `${process.env.API_URL}/webhook`;

export const triggerYoutubeVideoScrape = async (url) => {
  const data = JSON.stringify([{ url, country: '' }]);

  const response = await fetch(
    `${brightDataTriggerUrl}?dataset_id=gd_lk56epmy2i5g7lzu0k&endpoint=${webhookUrl}&format=json&uncompressed_webhook=true&include_errors=true`,
    {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${process.env.BRIGHTDATA_API_KEY}`,
        'Content-Type': 'application/json',
      },
      body: data,
    }
  );
  const result = await response.json();
  console.log(result);
  return result.snapshot_id;
};
